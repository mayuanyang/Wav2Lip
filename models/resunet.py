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
            self.blocks.append(ProcessBlock(12))
        
        print('The length of blocks', len(self.blocks))
        self.output_block = ProcessBlock(3)

        self.face_enhancer = FaceEnhancer()
                
        
    def forward(self, audio_sequences, face_sequences):
        temp_output = None
        face_input = face_sequences
        for block in self.blocks:
            if temp_output is not None:
                face_input = face_sequences + temp_output
            
            temp_output = self.forward_impl(audio_sequences, face_input, block.face_encoder_blocks, block.audio_encoder, block.face_decoder_blocks, block.attention_blocks, block.output_block)
            
        step2_face_sequences = face_sequences + temp_output
        outputs = self.forward_impl(audio_sequences, step2_face_sequences, self.output_block.face_encoder_blocks, self.output_block.audio_encoder, self.output_block.face_decoder_blocks, self.output_block.attention_blocks, self.output_block.output_block)

        
        # Select the last 3 groups from face_sequences
        input_dim_size = len(face_sequences.size())
        if input_dim_size > 4:
          group0 = face_sequences[:, 0:3, :, :, :]
          group1 = face_sequences[:, 3:6, :, :, :]
          group2 = face_sequences[:, 6:9, :, :, :]
          group3 = face_sequences[:, 9:12, :, :, :]
        else:
          group0 = face_sequences[:, 0:3, :, :]
          group1 = face_sequences[:, 3:6, :, :]
          group2 = face_sequences[:, 6:9, :, :]
          group3 = face_sequences[:, 9:12, :, :]  

        # Progressive residual connections
        temp_output = group0 + group1  # First residual connection
        temp_output = temp_output + group2  # First residual connection
        temp_output = temp_output + group3  # Second residual connection

        # Perform the residual connection
        x = temp_output + outputs

        outputs = self.face_enhancer(x)
      
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
            nn.Sequential(Conv2d(12, 96, kernel_size=3, stride=1, padding=1), #1+(3−1)×1=3
                          Conv2d(96, 96, kernel_size=3, stride=1, padding=1, residual=True), #5
                          ), # 192,192

            nn.Sequential(Conv2d(96, 96, kernel_size=3, stride=2, padding=1), #5+(3−1)×2=9
              Conv2d(96, 96, kernel_size=3, stride=1, padding=1, residual=True), #11
              ), # 96,96

            nn.Sequential(Conv2d(96, 96, kernel_size=3, stride=2, padding=1), # 48,48, 11+(3−1)×2=15
            Conv2d(96, 96, kernel_size=3, stride=1, padding=1, residual=True), #17
            ),

            nn.Sequential(Conv2d(96, 96, kernel_size=3, stride=2, padding=1), # 24,24, 17+(3−1)×2=21
            Conv2d(96, 96, kernel_size=3, stride=1, padding=1, residual=True), #23
            ),

            nn.Sequential(Conv2d(96, 128, kernel_size=3, stride=2, padding=1), # 12,12, 23+(3−1)×2=27
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True) # 29
            ), 

            nn.Sequential(Conv2d(128, 128, kernel_size=3, stride=2, padding=1), # 6,6, 29+(3−1)×2=33
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True)), #35

            nn.Sequential(Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # 3,3, 35+(3−1)×2=39
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),), #41
            
            nn.Sequential(Conv2d(256, 256, kernel_size=3, stride=1, padding=0), # 1, 1, 41+(3−1)×1=43
            Conv2d(256, 256, kernel_size=1, stride=1, padding=0, residual=True))]) # 45

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

            nn.Sequential(Conv2dTranspose(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1), 
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            ), # 48, 48

            nn.Sequential(Conv2dTranspose(224, 96, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(96, 96, kernel_size=3, stride=1, padding=1, residual=True),
            ), # 96,96
            
            nn.Sequential(
                Conv2dTranspose(192, 96, kernel_size=3, stride=2, padding=1, output_padding=1),
                Conv2d(96, 96, kernel_size=3, stride=1, padding=1, residual=True),
            )]) 

        self.output_block = nn.Sequential(Conv2d(192, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, output_block_channels, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid())
        
        # Define attention gates corresponding to each skip connection
        # Adjust F_g and F_l based on your architecture's channel dimensions
        self.attention_blocks = nn.ModuleList([
            AttentionGate(F_g=256, F_l=256, F_int=128),
            AttentionGate(F_g=256, F_l=256, F_int=128),
            AttentionGate(F_g=256, F_l=128, F_int=128),
            AttentionGate(F_g=192, F_l=128, F_int=128),
            AttentionGate(F_g=160, F_l=96, F_int=128),
            AttentionGate(F_g=128, F_l=96, F_int=128),
            AttentionGate(F_g=96, F_l=96, F_int=128),
            AttentionGate(F_g=96, F_l=96, F_int=128),
            # Add more attention gates if you have more skip connections
        ])

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

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.relu(self.norm(self.conv1(x)))
        x = self.relu(self.norm(self.conv2(x)))
        return x

class FaceEnhancer(nn.Module):
    def __init__(self):
        super(FaceEnhancer, self).__init__()
        
        # Downsampling blocks
        self.enc1 = UNetBlock(3, 32)
        self.enc2 = UNetBlock(32, 64)
        self.enc3 = UNetBlock(64, 128)
        self.enc4 = UNetBlock(128, 256)
        
        # Bottleneck
        self.bottleneck = UNetBlock(256, 512)
        
        # Upsampling blocks
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = UNetBlock(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = UNetBlock(256, 128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = UNetBlock(128, 64)
        
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = UNetBlock(64, 32)
        
        # Final output layer
        self.final_conv = nn.Conv2d(32, 3, kernel_size=1)
        # self.final_conv = nn.Sequential(Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
        #     nn.Sigmoid())
    
    def forward(self, x):
        # Reshape to combine B and T for processing in U-Net
        B = x.size(0)
        input_dim_size = len(x.size())

        if input_dim_size > 4:
            x = torch.cat([x[:, :, i] for i in range(x.size(2))], dim=0)
        
        # Downsampling path
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))
        
        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))
        
        # Upsampling path
        dec4 = self.dec4(torch.cat([self.up4(bottleneck), enc4], dim=1))
        dec3 = self.dec3(torch.cat([self.up3(dec4), enc3], dim=1))
        dec2 = self.dec2(torch.cat([self.up2(dec3), enc2], dim=1))
        dec1 = self.dec1(torch.cat([self.up1(dec2), enc1], dim=1))
        
        # Final output layer
        x = self.final_conv(dec1)

        if input_dim_size > 4:
            x = torch.split(x, B, dim=0) # [(B, C, H, W)]
            outputs = torch.stack(x, dim=2) # (B, C, T, H, W)
        else:
            outputs = x
        
        return outputs
