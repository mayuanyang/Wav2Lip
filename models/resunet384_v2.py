import torch
from torch import nn
from torch.nn import functional as F

from .conv import Conv2dTranspose, Conv2d, nonorm_Conv2d

class CrossModalAttention2d(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(CrossModalAttention2d, self).__init__()
        # Project face features into a lower-dimensional query space
        self.query_conv = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        # Project audio features into key space
        self.key_conv = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        # Project audio features into value space (we keep full channel dimension)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        # Learnable scaling factor
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, face_feat, audio_feat):
        """
        face_feat: Tensor of shape [B, C, H, W] from the face bottleneck
        audio_feat: Tensor of shape [B, C, H, W] from the audio encoder
        """
        B, C, H, W = face_feat.size()
        # Compute query from face features
        query = self.query_conv(face_feat).view(B, -1, H * W).permute(0, 2, 1)  # (B, HW, C//reduction)
        # Compute key from audio features
        key = self.key_conv(audio_feat).view(B, -1, H * W)                       # (B, C//reduction, HW)
        # Dot-product to compute attention map
        energy = torch.bmm(query, key)                                           # (B, HW, HW)
        attention = F.softmax(energy, dim=-1)                                    # (B, HW, HW)
        # Compute value from audio features
        value = self.value_conv(audio_feat).view(B, -1, H * W)                   # (B, C, HW)
        # Aggregate audio features using the attention map
        out = torch.bmm(value, attention.permute(0, 2, 1))                       # (B, C, HW)
        out = out.view(B, C, H, W)
        # Residual connection
        out = self.gamma * out + face_feat
        return out

class ResUNet384V2(nn.Module):
    def __init__(self, num_of_blocks=2):
        super(ResUNet384V2, self).__init__()
        self.face_encoder1 = nn.Sequential(
            Conv2d(12, 64, kernel_size=7, stride=1, padding=3),
            Conv2d(64, 64, kernel_size=7, stride=1, padding=3, residual=True),
            Conv2d(64, 64, kernel_size=7, stride=1, padding=3, residual=True),
        )
        self.fe_down1 = nn.Sequential(
            Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
        )
        
        self.face_encoder2 = nn.Sequential(
            Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
        )
        self.fe_down2 = nn.Sequential(
            Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
        )

        self.face_encoder3 = nn.Sequential(
            Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
        )
        self.fe_down3 = nn.Sequential(
            Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
        )

        self.face_encoder4 = nn.Sequential(
            Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
        )
        self.fe_down4 = nn.Sequential(
            Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
        )

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            
            Conv2d(32, 64, kernel_size=3, stride=(2, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 64, kernel_size=3, stride=(2, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=1, padding=(0, 1)),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2dTranspose(256, 256, kernel_size=3, stride=3, padding=0),
            Conv2d(256, 1024, kernel_size=1, stride=1, padding=0)
        ) 

        self.bottlenet = nn.Sequential(
            Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
        )
        
        # Replace the simple fusion with cross-modal attention:
        self.cross_modal_attention = CrossModalAttention2d(1024, reduction=8)
        
        # Decoder with cross-attention
        self.face_decoder4 = nn.Sequential(
            Conv2dTranspose(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
        )
        self.fd_conv4 = nn.Sequential(
            Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
        )

        self.face_decoder3 = nn.Sequential(
            Conv2dTranspose(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
        )
        self.fd_conv3 = nn.Sequential(
            Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
        )
        

        self.face_decoder2 = nn.Sequential(
            Conv2dTranspose(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
        )
        self.fd_conv2 = nn.Sequential(
            Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
        )
        

        self.face_decoder1 = nn.Sequential(
            Conv2dTranspose(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
        )
        
        self.fd_conv1 = nn.Sequential(
            Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
        )

        self.output_block = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, audio_sequences, face_sequences):
        def check_nan(tensor, name):
            if torch.isnan(tensor).any():
                print('NaN problem', f"NaN in {name}") 
         
        B = audio_sequences.size(0)
        input_dim_size = len(face_sequences.size())
        
        if input_dim_size > 4:
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)

        # Obtain audio features
        audio_embedding = self.audio_encoder(audio_sequences)
        # Process face images through the encoder
        face1 = self.face_encoder1(face_sequences)
        fed1 = self.fe_down1(face1)

        face2 = self.face_encoder2(fed1)
        fed2 = self.fe_down2(face2)

        face3 = self.face_encoder3(fed2)
        fed3 = self.fe_down3(face3)

        face4 = self.face_encoder4(fed3)
        fed4 = self.fe_down4(face4)
        
        # Get face bottleneck features
        bottlenet = self.bottlenet(fed4)
        # Instead of simply adding the audio_embedding, perform cross-modal attention.
        # Here, we use the face bottleneck as queries and the audio_embedding as keys/values.
        bottlenet = self.cross_modal_attention(bottlenet, audio_embedding)

        deface4 = self.face_decoder4(bottlenet)
        cat4 = torch.cat([deface4, face4], dim=1)
        cat4 = self.fd_conv4(cat4)

        deface3 = self.face_decoder3(cat4)
        cat3 = torch.cat([deface3, face3], dim=1)
        cat3 = self.fd_conv3(cat3)

        deface2 = self.face_decoder2(cat3)
        cat2 = torch.cat([deface2, face2], dim=1)
        cat2 = self.fd_conv2(cat2)
        
        deface1 = self.face_decoder1(cat2)
        cat1 = torch.cat([deface1, face1], dim=1)
        cat1 = self.fd_conv1(cat1)

        x = self.output_block(cat1)

        if input_dim_size > 4:
            x = torch.split(x, B, dim=0)
            outputs = torch.stack(x, dim=2)
        else:
            outputs = x
            
        return outputs
