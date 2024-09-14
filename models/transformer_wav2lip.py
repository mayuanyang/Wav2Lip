import torch
from torch import nn
from torch.nn import functional as F
import math

from .conv import Conv2dTranspose, Conv2d, nonorm_Conv2d

class TransformerWav2Lip(nn.Module):
    def __init__(self):
        super(TransformerWav2Lip, self).__init__()

        '''
        Outpu = Input + (k-1) x S

        Where:
        Input is the receptive field size from the previous layer.
        k is the kernel size.
        S is the stride.
        '''

        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(6, 32, kernel_size=7, stride=1, padding=3), #1+(7−1)×1=7
                          Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True), #9
                          Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True), #11
                          ), # 192,192

            nn.Sequential(Conv2d(32, 64, kernel_size=7, stride=2, padding=3), #11+(7−1)×2=23
              Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), #25
              Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), #27
              ), # 96,96

            nn.Sequential(Conv2d(64, 128, kernel_size=7, stride=2, padding=3), # 48,48, 27+(7−1)×2=39
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), #41
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), #43
            ),

            nn.Sequential(Conv2d(128, 128, kernel_size=7, stride=2, padding=3), # 24,24, 43+(7−1)×2=55
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), #57
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), #59
            ),

            nn.Sequential(Conv2d(128, 256, kernel_size=7, stride=2, padding=3), # 12,12, 59+(7−1)×2=71
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True), #73
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True)), #75

            nn.Sequential(Conv2d(256, 512, kernel_size=5, stride=2, padding=2), # 6,6, 75+(5−1)×2=83
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True)), #85

            nn.Sequential(Conv2d(512, 512, kernel_size=5, stride=2, padding=2), # 3,3, 85+(5−1)×2=93
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),), #95
            
            nn.Sequential(Conv2d(512, 512, kernel_size=3, stride=1, padding=0), # 1, 1, 95+(3−1)×1=97
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0)),])

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
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

        self.face_decoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),),

            nn.Sequential(Conv2dTranspose(1024, 768, kernel_size=3, stride=1, padding=0), # 3,3
            Conv2d(768, 768, kernel_size=3, stride=1, padding=1, residual=True),),

            nn.Sequential(Conv2dTranspose(1280, 640, kernel_size=2, stride=2, padding=0), # 6, 6
            Conv2d(640, 640, kernel_size=1, stride=1, padding=0, residual=True),
            Conv2d(640, 640, kernel_size=1, stride=1, padding=0, residual=True),), 

            nn.Sequential(Conv2dTranspose(1152, 512, kernel_size=2, stride=2, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0, residual=True),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0, residual=True),), # 12, 12

            nn.Sequential(Conv2dTranspose(768, 256, kernel_size=2, stride=2, padding=0), # 24, 24
            Conv2d(256, 256, kernel_size=1, stride=1, padding=0, residual=True),
            Conv2d(256, 256, kernel_size=1, stride=1, padding=0, residual=True),), 

            nn.Sequential(Conv2dTranspose(384, 128, kernel_size=2, stride=2, padding=0), 
            Conv2d(128, 128, kernel_size=2, stride=1, padding=1, dilation=2, residual=True),
            Conv2d(128, 128, kernel_size=2, stride=1, padding=1, dilation=2, residual=True),
            ), # 48, 48

            nn.Sequential(Conv2dTranspose(256, 64, kernel_size=2, stride=2, padding=0),
            Conv2d(64, 64, kernel_size=2, stride=1, padding=1, dilation=2, residual=True),
            ), # 96,96
            
            nn.Sequential(
                Conv2dTranspose(128, 64, kernel_size=2, stride=2, padding=0),
                Conv2d(64, 64, kernel_size=2, stride=1, padding=1, dilation=2),
                
            )]) 

        self.output_block = nn.Sequential(Conv2d(96, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid())
        
    def forward(self, audio_sequences, face_sequences):
        # audio_sequences = (B, T, 1, 80, 16)
        B = audio_sequences.size(0)

        input_dim_size = len(face_sequences.size())
        if input_dim_size > 4:
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)

        audio_embedding = self.audio_encoder(audio_sequences) # B, 512, 1, 1

        face_features = []
        this_face_sequence = face_sequences
        for f in self.face_encoder_blocks:
            this_face_sequence = f(this_face_sequence)
            face_features.append(this_face_sequence)

        x = audio_embedding
        index = 1
        for f in self.face_decoder_blocks:
            #Use the face decoder to decode the audio
            x = f(x)
            index += 1

            try:
                '''
                Concat the decoded audio with the correspondent face features, 
                this also known as skip connection by concatinationg the encoder's face feature and decoded audio features
                '''
                x = torch.cat((x, face_features[-1]), dim=1)
            except Exception as e:
                raise e
            
            face_features.pop()
            #print('The new length', len(feats))

        '''
        Eddy: We might want to use a transformer to learn the combined audio and face features rather than using the concatenation 
        of the decoded audio and face features
        '''
        
        # Try to do transformer here with x and audio embedding

        # x is the combined audio and face features
        x = self.output_block(x)

        if input_dim_size > 4:
            x = torch.split(x, B, dim=0) # [(B, C, H, W)]
            outputs = torch.stack(x, dim=2) # (B, C, T, H, W)

        else:
            outputs = x
            
        return outputs
