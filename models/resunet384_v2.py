import torch
from torch import nn
from torch.nn import functional as F

from .conv import Conv2dTranspose, Conv2d, nonorm_Conv2d

class CrossAttention(nn.Module):
    def __init__(self, in_channels, audio_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = in_channels // num_heads
        self.scale = self.dim_head ** -0.5

        self.audio_to_kv = nn.Linear(audio_dim, 2 * in_channels)
        self.face_to_q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x, audio_embedding):
        B, C, H, W = x.shape
        audio_embedding = audio_embedding.view(B, -1)
        
        kv = self.audio_to_kv(audio_embedding)
        k, v = kv.chunk(2, dim=1)
        
        q = self.face_to_q(x)
        q = q.reshape(B, self.num_heads, self.dim_head, H * W).permute(0, 1, 3, 2)
        
        k = k.reshape(B, self.num_heads, self.dim_head, 1).permute(0, 1, 3, 2)
        v = v.reshape(B, self.num_heads, self.dim_head, 1).permute(0, 1, 3, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        out = self.proj(out)
        
        return x + out

class ResUNet384V2(nn.Module):
    def __init__(self, num_of_blocks=2):
        super(ResUNet384, self).__init__()
        self.face_encoder1 = nn.Sequential(
            Conv2d(12, 64, kernel_size=7, stride=1, padding=3),
            Conv2d(64, 64, kernel_size=7, stride=1, padding=3, residual=True),
        )
        self.fe_down1 = Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        
        self.face_encoder2 = nn.Sequential(
            Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
        )
        self.fe_down2 = Conv2d(128, 128, kernel_size=3, stride=2, padding=1)

        self.face_encoder3 = nn.Sequential(
            Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
        )
        self.fe_down3 = Conv2d(256, 256, kernel_size=3, stride=2, padding=1)

        self.face_encoder4 = nn.Sequential(
            Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
        )
        self.fe_down4 = Conv2d(512, 512, kernel_size=3, stride=2, padding=1)

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0, residual=True),
        )

        self.bottlenet = nn.Sequential(
            Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
        )
        
        # Decoder with cross-attention
        self.face_decoder4 = nn.Sequential(
            Conv2dTranspose(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        )
        self.fd_conv4 = Conv2d(1024, 512, kernel_size=3, stride=1, padding=1)
        self.cross_attn4 = CrossAttention(512, 512)

        self.face_decoder3 = nn.Sequential(
            Conv2dTranspose(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        )
        self.fd_conv3 = Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.cross_attn3 = CrossAttention(256, 512)

        self.face_decoder2 = nn.Sequential(
            Conv2dTranspose(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
        )
        self.fd_conv2 = Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.cross_attn2 = CrossAttention(128, 512)

        self.face_decoder1 = nn.Sequential(
            Conv2dTranspose(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
        )
        self.cross_attn1 = CrossAttention(64, 512)

        self.dropout = nn.Dropout(0.1)

        self.output_block = nn.Sequential(nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0))

    def forward(self, audio_sequences, face_sequences):
        B = audio_sequences.size(0)
        input_dim_size = len(face_sequences.size())
        
        if input_dim_size > 4:
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)

        audio_embedding = self.audio_encoder(audio_sequences)

        face1 = self.face_encoder1(face_sequences)
        fed1 = self.fe_down1(face1)

        face2 = self.face_encoder2(fed1)
        fed2 = self.fe_down2(face2)

        face3 = self.face_encoder3(fed2)
        fed3 = self.fe_down3(face3)

        face4 = self.face_encoder4(fed3)
        fed4 = self.fe_down4(face4)
        
        bottlenet = self.bottlenet(fed4)

        deface4 = self.face_decoder4(bottlenet)
        deface4 = self.cross_attn4(deface4, audio_embedding)
        cat4 = torch.cat([deface4, face4], dim=1)
        cat4 = self.fd_conv4(cat4)

        deface3 = self.face_decoder3(cat4)
        deface3 = self.cross_attn3(deface3, audio_embedding)
        cat3 = torch.cat([deface3, face3], dim=1)
        cat3 = self.fd_conv3(cat3)

        deface2 = self.face_decoder2(cat3)
        deface2 = self.cross_attn2(deface2, audio_embedding)
        cat2 = torch.cat([deface2, face2], dim=1)
        cat2 = self.fd_conv2(cat2)

        deface1 = self.face_decoder1(cat2)
        deface1 = self.cross_attn1(deface1, audio_embedding)
        
        x = self.output_block(deface1)

        x = torch.sigmoid(x)
        
        x = self.dropout(x)

        if input_dim_size > 4:
            x = torch.split(x, B, dim=0)
            outputs = torch.stack(x, dim=2)
        else:
            outputs = x
            
        return outputs