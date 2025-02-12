import torch
from torch import nn
from torch.nn import functional as F

from .conv import Conv2dTranspose, Conv2d, nonorm_Conv2d


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

        self.face_audio_learner = nn.Sequential(
            Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, residual=True),
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
        bottlenet = self.face_audio_learner(bottlenet)
        check_nan(bottlenet, "bottlenet")

        deface4 = self.face_decoder4(bottlenet)
        check_nan(deface4, "deface4")
        
        check_nan(deface4, "cross_attn4")
        cat4 = torch.cat([deface4, face4], dim=1)
        cat4 = self.fd_conv4(cat4)
        check_nan(cat4, "fd_conv4")

        deface3 = self.face_decoder3(cat4)
        check_nan(deface3, "face_decoder3")
        
        cat3 = torch.cat([deface3, face3], dim=1)
        cat3 = self.fd_conv3(cat3)
        check_nan(cat3, "fd_conv3")

        deface2 = self.face_decoder2(cat3)
        check_nan(deface2, "face_decoder2")
        
        cat2 = torch.cat([deface2, face2], dim=1)
        cat2 = self.fd_conv2(cat2)
        check_nan(cat2, "fd_conv2")
        
        deface1 = self.face_decoder1(cat2)
        check_nan(deface1, "face_decoder1")

        cat1 = torch.cat([deface1, face1], dim=1)
        cat1 = self.fd_conv1(cat1)
        check_nan(cat1, "fd_conv1")

        x = self.output_block(cat1)
        check_nan(x, "output_block")

        if input_dim_size > 4:
            x = torch.split(x, B, dim=0)
            outputs = torch.stack(x, dim=2)
        else:
            outputs = x
            
        return outputs