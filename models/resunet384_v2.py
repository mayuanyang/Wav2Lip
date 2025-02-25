import torch
import math
from torch import nn
from torch.nn import functional as F

from .conv import Conv2dTranspose, Conv2d, nonorm_Conv2d
from .cross_modal_attention import CrossModalAttention2d
from .self_attention import AttentionBlock


def check_nan(tensor, name):
  if torch.isnan(tensor).any():
    print('NaN problem', f"NaN in {name}") 
    
class ResUNet384V2(nn.Module):
    def __init__(self):
        super(ResUNet384V2, self).__init__()
        
        self.face_gt_bottom_encoder = nn.Sequential( # H W 192x384
            Conv2d(12, 64, kernel_size=7, stride=1, padding=3),
            Conv2d(64, 64, kernel_size=7, stride=1, padding=3, residual=True),
            Conv2d(64, 64, kernel_size=7, stride=1, padding=3, residual=True),
            
            Conv2d(64, 64, kernel_size=3, stride=2, padding=1), # H W 96x192
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
        
        self.face_encoder1 = nn.Sequential( #384x384
            Conv2d(12, 64, kernel_size=7, stride=1, padding=3),
            Conv2d(64, 64, kernel_size=7, stride=1, padding=3, residual=True),
            Conv2d(64, 64, kernel_size=7, stride=1, padding=3, residual=True),
        )
        self.fe_down1 = nn.Sequential(
            Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
        )
        
        self.face_gt_attn = AttentionBlock(64, reduction=1)
        #self.face1_attn = AttentionBlock(64, reduction=2, sparse_attention=True, window_size=25)
        
        self.face_encoder2 = nn.Sequential( #192x192
            Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
        )
        self.fe_down2 = nn.Sequential(
            Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
        )

        self.face_encoder3 = nn.Sequential( #96x96
            Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
        )
        self.fe_down3 = nn.Sequential(
            Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
        )

        self.face_encoder4 = nn.Sequential( #48x48
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

        self.bottlenet = nn.Sequential(
            Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
        )
        
        self.cross_modal_attention_gt = CrossModalAttention2d(64, reduction=8)
        
        # Decoder with cross-attention
        self.face_decoder4 = nn.Sequential( #48x48
            Conv2dTranspose(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
        )
        self.fd_conv4 = nn.Sequential(
            Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
        )
        
        #self.deface4_attn = AttentionBlock(512)

        self.face_decoder3 = nn.Sequential( #96x96
            Conv2dTranspose(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
        )
        self.fd_conv3 = nn.Sequential(
            Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
        )
        

        self.face_decoder2 = nn.Sequential( #192x192
            Conv2dTranspose(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
        )
        self.fd_conv2 = nn.Sequential(
            Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
        )


        self.face_decoder1 = nn.Sequential( #384x384
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
        
        bottom_half_face = face_sequences[:, :, H//2:, :]
        
        bottom_face = self.face_gt_bottom_encoder(bottom_half_face)
        bottom_face = self.face_gt_attn(bottom_face)
               
        # Obtain audio features
        audio_embedding1 = self.audio_encoder1(audio_sequences)
        
        gt_attn = self.cross_modal_attention_gt(bottom_face, audio_embedding1)
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

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # Input channels are now 3
            Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            Conv2d(512, 1, kernel_size=4, stride=1, padding=1)  # Output a single score
        )

    def forward(self, x):
        input_dim_size = len(x.size())
        if input_dim_size > 4:
            x = torch.cat([x[:, :, i] for i in range(x.size(2))], dim=0)
        return self.model(x)