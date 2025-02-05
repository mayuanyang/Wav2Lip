import torch
from torch import nn
from torch.nn import functional as F

from .conv import Conv2dTranspose, Conv2d, nonorm_Conv2d

class CrossAttention(nn.Module):
    def __init__(self, in_channels, audio_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = in_channels // num_heads

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
        
        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.dim_head ** 0.5)
        attn = attn.softmax(dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        out = self.proj(out)
        
        return x + out * 0.1

class ResUNet384V2(nn.Module):
    def __init__(self, num_of_blocks=2):
        super(ResUNet384V2, self).__init__()
        self.face_encoder1 = nn.Sequential(
            Conv2d(12, 64, kernel_size=7, stride=1, padding=3),
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
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        )

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1), # 80x16, 1+(3−1)×1=3
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True), # 3+(3-1)x1=5
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True), # 5+(3-1)x1=7
            
            Conv2d(32, 64, kernel_size=3, stride=(2, 1), padding=1), # 40x16, 11+(3-1)x3=17
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), #15+(3-1)x1=19
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), #17+(3-1)x1=21

            Conv2d(64, 64, kernel_size=3, stride=(2, 1), padding=1), # 20x16, 25+(3-1)x2=29
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), #29+(3-1)x1=31

            Conv2d(64, 128, kernel_size=3, stride=2, padding=1), #10x8, 35+(3-1)x2=41
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), #41+(3-1)x1=43

            Conv2d(128, 256, kernel_size=3, stride=1, padding=(0 ,1)), # 8x8 No padding; adjust based on your need for feature extraction.
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True), #51+(3-1)x1=53

            Conv2dTranspose(256, 256, kernel_size=3, stride=3, padding=0),
            Conv2d(256, 1024, kernel_size=1, stride=1, padding=0)
        ) 

        self.bottlenet = nn.Sequential(
            Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
        )
        
        # Decoder with cross-attention
        self.face_decoder4 = nn.Sequential(
            Conv2dTranspose(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
        )
        self.fd_conv4 = nn.Sequential(
            Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
        )
        #self.cross_attn4 = CrossAttention(512, 512)

        self.face_decoder3 = nn.Sequential(
            Conv2dTranspose(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
        )
        self.fd_conv3 = nn.Sequential(
            Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
        )
        #self.cross_attn3 = CrossAttention(256, 512)

        self.face_decoder2 = nn.Sequential(
            Conv2dTranspose(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
        )
        self.fd_conv2 = nn.Sequential(
            Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
        )
        #self.cross_attn2 = CrossAttention(128, 512)

        self.face_decoder1 = nn.Sequential(
            Conv2dTranspose(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
        )
        #self.cross_attn1 = CrossAttention(64, 512)

        self.output_block = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid())

    def forward(self, audio_sequences, face_sequences):
        def check_nan(tensor, name):
          if torch.isnan(tensor).any():
              print('NaN problem', f"NaN in {name}") 
          

         
        B = audio_sequences.size(0)
        input_dim_size = len(face_sequences.size())
        
        if input_dim_size > 4:
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)

        audio_embedding = self.audio_encoder(audio_sequences)

        face1 = self.face_encoder1(face_sequences)
        check_nan(face1, "face_encoder1")
        fed1 = self.fe_down1(face1)
        check_nan(fed1, "fed1")

        face2 = self.face_encoder2(fed1)
        check_nan(face2, "face2")
        fed2 = self.fe_down2(face2)
        check_nan(fed2, "fed2")

        face3 = self.face_encoder3(fed2)
        check_nan(face3, "face3")
        fed3 = self.fe_down3(face3)
        check_nan(fed3, "fed3")

        face4 = self.face_encoder4(fed3)
        check_nan(face4, "face4")
        fed4 = self.fe_down4(face4)
        check_nan(fed4, "fed4")
        
        bottlenet = self.bottlenet(fed4)
        bottlenet = bottlenet + audio_embedding
        check_nan(bottlenet, "bottlenet")

        deface4 = self.face_decoder4(bottlenet)
        check_nan(deface4, "deface4")
        #deface4 = self.cross_attn4(deface4, audio_embedding)
        check_nan(deface4, "cross_attn4")
        cat4 = torch.cat([deface4, face4], dim=1)
        cat4 = self.fd_conv4(cat4)
        check_nan(cat4, "fd_conv4")

        deface3 = self.face_decoder3(cat4)
        check_nan(deface3, "face_decoder3")
        #deface3 = self.cross_attn3(deface3, audio_embedding)
        cat3 = torch.cat([deface3, face3], dim=1)
        cat3 = self.fd_conv3(cat3)
        check_nan(cat3, "fd_conv3")

        deface2 = self.face_decoder2(cat3)
        check_nan(deface2, "face_decoder2")
        #deface2 = self.cross_attn2(deface2, audio_embedding)
        cat2 = torch.cat([deface2, face2], dim=1)
        cat2 = self.fd_conv2(cat2)
        check_nan(cat2, "fd_conv2")

        deface1 = self.face_decoder1(cat2)
        check_nan(deface1, "face_decoder1")
        #deface1 = self.cross_attn1(deface1, audio_embedding)
        
        x = self.output_block(deface1)
        check_nan(x, "output_block")

        if input_dim_size > 4:
            x = torch.split(x, B, dim=0)
            outputs = torch.stack(x, dim=2)
        else:
            outputs = x
            
        return outputs