import torch
from torch import nn
from torch.nn import functional as F
import math

from .conv import Conv2dTranspose, Conv2d, nonorm_Conv2d

class ResUNet(nn.Module):
    def __init__(self, num_of_blocks=2):
        super(ResUNet, self).__init__()
        self.blocks = nn.ModuleList()

        for i in range(num_of_blocks):
            self.blocks.append(ProcessBlock(9))
        
        print('The length of blocks', len(self.blocks))
        self.output_block = ProcessBlock(3)
                
        
    def forward(self, audio_sequences, face_sequences):
        temp_output = None
        face_input = face_sequences
        for block in self.blocks:
            if temp_output is not None:
                face_input = face_sequences + temp_output
            
            temp_output = self.forward_impl(audio_sequences, face_input, block.face_encoder_blocks, block.audio_encoder, block.face_decoder_blocks, block.attention_blocks, block.output_block)
            
        step2_face_sequences = face_sequences + temp_output
        outputs = self.forward_impl(audio_sequences, step2_face_sequences, self.output_block.face_encoder_blocks, self.output_block.audio_encoder, self.output_block.face_decoder_blocks, self.output_block.attention_blocks, self.output_block.output_block)
      
        return outputs

    def forward_impl(self, audio_sequences, face_sequences, face_encoder_blocks, audio_encoder, face_decoder_blocks, cross_att_blocks, output_block):
        # audio_sequences = (B, T, 1, 80, 16)
        B = audio_sequences.size(0)

        input_dim_size = len(face_sequences.size())
        if input_dim_size > 4:
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)

        audio_embedding = audio_encoder(audio_sequences) # B, 512, 1, 1

        face_features = []
        this_face_sequence = face_sequences
        for f in face_encoder_blocks:
            this_face_sequence = f(this_face_sequence)
            face_features.append(this_face_sequence)

        # NeRF-enhanced decoding
        x = audio_embedding
        # Apply attention before concatenation
        for i, f in enumerate(face_decoder_blocks):
            x = f(x)
            if face_features:
                skip = face_features.pop()
                skip = cross_att_blocks[i](x, skip)
                x = torch.cat((x, skip), dim=1)

        x = output_block(x)

        if input_dim_size > 4:
            
            x = torch.split(x, B, dim=0) # [(B, C, H, W)]
            
            outputs = torch.stack(x, dim=2) # (B, C, T, H, W)

        else:
            outputs = x
            
        return outputs


class ProcessBlock(nn.Module):
    def __init__(self, output_block_channels) -> None:
        super(ProcessBlock, self).__init__()
        '''
        Outpu = Input + (k-1) x S

        Where:
        Input is the receptive field size from the previous layer.
        k is the kernel size.
        S is the stride.
        '''

        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(9, 64, kernel_size=7, stride=1, padding=3), #1+(7−1)×1=7
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), #9
                          ), # 192,192

            nn.Sequential(Conv2d(64, 64, kernel_size=7, stride=2, padding=3), #11+(7−1)×2=23
              Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), #25
              ), # 96,96

            nn.Sequential(Conv2d(64, 64, kernel_size=7, stride=2, padding=3), # 48,48, 27+(7−1)×2=39
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), #41
            ),

            nn.Sequential(Conv2d(64, 64, kernel_size=7, stride=2, padding=3), # 24,24, 43+(7−1)×2=55
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), #57
            ),

            nn.Sequential(Conv2d(64, 128, kernel_size=7, stride=2, padding=3), # 12,12, 59+(7−1)×2=71
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True)
            ), #73

            nn.Sequential(Conv2d(128, 128, kernel_size=5, stride=2, padding=2), # 6,6, 75+(5−1)×2=83
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True)), #85

            nn.Sequential(Conv2d(128, 256, kernel_size=5, stride=2, padding=2), # 3,3, 85+(5−1)×2=93
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),), #95
            
            nn.Sequential(Conv2d(256, 256, kernel_size=3, stride=1, padding=0), # 1, 1, 95+(3−1)×1=97
            Conv2d(256, 256, kernel_size=1, stride=1, padding=0, residual=True))])

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0, residual=True),)

        self.face_decoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
                          Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),),

            nn.Sequential(Conv2dTranspose(512, 256, kernel_size=3, stride=1, padding=0), # 3,3
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),),

            nn.Sequential(Conv2dTranspose(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),), # 6, 6

            nn.Sequential(Conv2dTranspose(384, 192, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(192, 192, kernel_size=3, stride=1, padding=1, residual=True),
            ), # 12, 12

            nn.Sequential(Conv2dTranspose(320, 160, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(160, 160, kernel_size=3, stride=1, padding=1, residual=True),
            ), # 24, 24

            nn.Sequential(Conv2dTranspose(224, 128, kernel_size=3, stride=2, padding=1, output_padding=1), 
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            ), # 48, 48

            nn.Sequential(Conv2dTranspose(192, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            ), # 96,96
            
            nn.Sequential(
                Conv2dTranspose(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            )]) 

        self.output_block = nn.Sequential(Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, output_block_channels, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid())
        
        # Define attention gates corresponding to each skip connection
        # Adjust F_g and F_l based on your architecture's channel dimensions
        self.attention_blocks = nn.ModuleList([
            AttentionGate(F_g=256, F_l=256, F_int=128),
            AttentionGate(F_g=256, F_l=256, F_int=128),
            AttentionGate(F_g=256, F_l=128, F_int=128),
            AttentionGate(F_g=192, F_l=128, F_int=128),
            AttentionGate(F_g=160, F_l=64, F_int=128),
            AttentionGate(F_g=128, F_l=64, F_int=128),
            AttentionGate(F_g=64, F_l=64, F_int=128),
            AttentionGate(F_g=64, F_l=64, F_int=128),
            # Add more attention gates if you have more skip connections
        ])
    
    def forward(self, face_features, audio_embedding):
        """
        Add NeRF for face feature enhancement in the decoder.
        """
        # Decode face features using the decoder
        x = audio_embedding
        for f in self.face_decoder_blocks:
            x = f(x)
        
        # Sample 3D points (for simplicity, assume we have 3D coordinates here)
        B, C, H, W = x.shape
        grid = self.create_3d_grid(B, H, W)  # Sample 3D points for NeRF
        rgb, density = self.nerf(grid)

        # Combine NeRF output with decoded face features
        x = torch.cat([x, rgb.permute(0, 3, 1, 2)], dim=1)  # Concatenate NeRF RGB with face features

        # Pass the combined features through the output block
        x = self.output_block(x)
        return x

    def create_3d_grid(self, B, H, W):
        """
        Create a 3D grid of coordinates to pass through NeRF.
        This is a simple grid for illustration. In practice, use real 3D sampling.
        """
        # Create a mesh grid in the range of [-1, 1] for x, y, z
        grid_x, grid_y = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W))
        grid_z = torch.linspace(0, 1, H)  # Sample z-axis points for simplicity
        grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)  # Shape (H, W, 3)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # Repeat for batch size
        return grid.to(torch.float32)

class NeRF(nn.Module):
    def __init__(self, depth=8, width=256):
        super(NeRF, self).__init__()
        self.depth = depth
        self.width = width

        layers = [nn.Linear(3, width), nn.ReLU()]
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())

        # Output layers for RGB color and density
        self.fc_rgb = nn.Linear(width, 3)  # Output RGB color
        self.fc_density = nn.Linear(width, 1)  # Output density

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # x: input 3D coordinates (B, N, 3) where N is the number of sampled points
        h = self.mlp(x)
        rgb = torch.sigmoid(self.fc_rgb(h))  # RGB color
        density = torch.relu(self.fc_density(h))  # Density value
        return rgb, density

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        """
        Args:
            F_g: Number of channels in the gating signal (decoder feature map).
            F_l: Number of channels in the skip connection (encoder feature map).
            F_int: Number of intermediate channels.
        """
        super(AttentionGate, self).__init__()
        self.W_g = nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True)
        self.W_x = nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, decoded_features, encoded_features):
        """
        Args:
            decoded_features: Decoder feature map (gating signal).
            encoded_features: Encoder feature map (skip connection).
        Returns:
            Weighted encoder feature map.
        """
        g1 = self.W_g(decoded_features)
        x1 = self.W_x(encoded_features)
        psi = self.relu(g1 + x1)
        psi = self.sigmoid(self.psi(psi))
        return encoded_features * psi
