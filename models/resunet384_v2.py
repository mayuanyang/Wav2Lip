import torch
import math
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
        # Learnable scaling factor, init to 0, but will learn as it goes
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, face_feat, audio_feat):
        """
        face_feat: Tensor of shape [B, C, H, W] from the face bottleneck
        audio_feat: Tensor of shape [B, C, H, W] from the audio encoder
        """
        FB, FC, FH, FW = face_feat.size()
        AB, AC, AH, AW = audio_feat.size()
        # Compute query from face features
        query = self.query_conv(face_feat).view(FB, -1, FH * FW).permute(0, 2, 1)  # (B, HW, C//reduction)
        check_nan(query, 'query')
        
        # Compute key from audio features
        key = self.key_conv(audio_feat).view(AB, -1, AH * AW)                       # (B, C//reduction, HW)
        check_nan(key, 'key')
        
        d_k = query.size(-1)
        energy = torch.bmm(query, key) / math.sqrt(d_k)
        energy = energy - energy.max(dim=-1, keepdim=True)[0]
        
        energy = energy.clamp(min=-50, max=50)
        attention = F.softmax(energy, dim=-1)
        
        check_nan(attention, 'attention')
        
        # Compute value from audio features
        value = self.value_conv(audio_feat).view(AB, -1, AH * AW)                   # (B, C, HW)
        check_nan(value, 'value')
        
        # Aggregate audio features using the attention map
        out = torch.bmm(value, attention.permute(0, 2, 1))                       # (B, C, HW)
        check_nan(out, 'out')
        
        out = out.view(FB, FC, FH, FW)
        # Residual connection
        out = self.gamma * out + face_feat
        return out
    
def check_nan(tensor, name):
  if torch.isnan(tensor).any():
    print('NaN problem', f"NaN in {name}") 
    
class ResUNet384V2(nn.Module):
    def __init__(self, num_of_blocks=2):
        super(ResUNet384V2, self).__init__()
        
        self.face_gt_bottom_encoder = nn.Sequential( # H W 192x384
            Conv2d(3, 32, kernel_size=7, stride=1, padding=3),
            Conv2d(32, 32, kernel_size=7, stride=1, padding=3, residual=True),
            Conv2d(32, 32, kernel_size=7, stride=1, padding=3, residual=True),
            
            Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # H W 96x192
            Conv2d(64, 64, kernel_size=7, stride=1, padding=3, residual=True),
            Conv2d(64, 64, kernel_size=7, stride=1, padding=3, residual=True),
            
            Conv2d(64, 64, kernel_size=3, stride=(1, 2), padding=1), # H W 96x96
            Conv2d(64, 64, kernel_size=7, stride=1, padding=3, residual=True),
            Conv2d(64, 64, kernel_size=7, stride=1, padding=3, residual=True),
            
            Conv2d(64, 64, kernel_size=3, stride=2, padding=1), # H W 48x48
            Conv2d(64, 64, kernel_size=7, stride=1, padding=3, residual=True),
            Conv2d(64, 64, kernel_size=7, stride=1, padding=3, residual=True),
        )
        
        self.face_gt_bottom_upconv = nn.Sequential(
            Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
        )
        
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

        self.audio_encoder1 = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            
            Conv2d(32, 64, kernel_size=3, stride=(2, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
        )

        self.audio_encoder2 = nn.Sequential(
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
        self.cross_modal_attention_gt = CrossModalAttention2d(64, reduction=8)
        # New cross-modal attention module in the decoder (using audio_embedding1, 64 channels)
        self.cross_modal_attention_dec = CrossModalAttention2d(64, reduction=1)
        
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

         
        B = audio_sequences.size(0)
        input_dim_size = len(face_sequences.size())
        
        if input_dim_size > 4:
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)

        
        _, _, H, W = face_sequences.shape
        
        gt_imgs = face_sequences[:, :3, : , :]
        bottom_half_face = gt_imgs[:, :, H//2:, :]
        
        bottom_face_gt = self.face_gt_bottom_encoder(bottom_half_face)
               
        # Obtain audio features
        audio_embedding1 = self.audio_encoder1(audio_sequences)
        audio_embedding2 = self.audio_encoder2(audio_embedding1)
        
        gt_attn = self.cross_modal_attention_gt(bottom_face_gt, audio_embedding1)
        gt_attn = self.face_gt_bottom_upconv(gt_attn)
        
        # Process face images through the encoder
        face1 = self.face_encoder1(face_sequences)
        
        fed1 = self.fe_down1(face1)

        face2 = self.face_encoder2(fed1)
        fed2 = self.fe_down2(face2)

        face3 = self.face_encoder3(fed2)
        fed3 = self.fe_down3(face3)
        
        fed3 = fed3 + gt_attn

        face4 = self.face_encoder4(fed3)
        fed4 = self.fe_down4(face4)
        
        # Get face bottleneck features
        bottlenet = self.bottlenet(fed4)
        # Instead of simply adding the audio_embedding, perform cross-modal attention.
        # Here, we use the face bottleneck as queries and the audio_embedding as keys/values.
        bottlenet = self.cross_modal_attention(bottlenet, audio_embedding2)

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
