import torch
from torch import nn
from torch.nn import functional as F
import math

from .conv import Conv2dTranspose, Conv2d, nonorm_Conv2d

class Wav2Lip(nn.Module):
    def __init__(self):
        super(Wav2Lip, self).__init__()

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

            nn.Sequential(Conv2d(32, 64, kernel_size=3, stride=2, padding=1), #11+(3−1)×2=15
              Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), #17
              Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), #19
              ), # 96,96

            nn.Sequential(Conv2d(64, 64, kernel_size=3, stride=1, padding=1), #19+(3−1)×2=21
              Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), #23
              Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), #25
              ), # 96,96

            nn.Sequential(Conv2d(64, 64, kernel_size=3, stride=2, padding=1), # 48,48, 25+(3−1)×2=29
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), #31
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), #33
            ),

            nn.Sequential(Conv2d(64, 64, kernel_size=3, stride=1, padding=1), # 48,48, 33+(3−1)×1=35
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), #37
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), #39
            ),

            nn.Sequential(Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 24,24, 39+(3−1)×2=43
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), #45
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), #47
            ),

            nn.Sequential(Conv2d(128, 128, kernel_size=3, stride=1, padding=1), # 24,24, 47+(3−1)×1=49
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), #51
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), #53
            ),

            nn.Sequential(Conv2d(128, 128, kernel_size=3, stride=2, padding=1), # 12,12, 53+(3−1)×2=57
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), #59
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True)), #61

            nn.Sequential(Conv2d(128, 128, kernel_size=3, stride=1, padding=1), # 12,12, 61+(3−1)×1=63
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), #65
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True)), #67

            nn.Sequential(Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # 6,6, 67+(3−1)×2=71
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True), #73
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True)), #75

            nn.Sequential(Conv2d(256, 256, kernel_size=3, stride=1, padding=1), # 6,6, 75+(3−1)×1=77
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True), #79
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True)), #81

            nn.Sequential(Conv2d(256, 256, kernel_size=3, stride=2, padding=1), # 3,3, 81+(3−1)×2=85
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True), #87
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),), #89

            nn.Sequential(Conv2d(256, 256, kernel_size=3, stride=1, padding=1), # 3,3, 89+(3−1)×1=91
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True), #93
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),), #95
            
            nn.Sequential(Conv2d(256, 512, kernel_size=3, stride=1, padding=0), # 1, 1, 95+(3−1)×1=97
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

            nn.Sequential(Conv2dTranspose(1024, 320, kernel_size=3, stride=1, padding=0), # 3,3
            Conv2d(320, 320, kernel_size=3, stride=1, padding=1, residual=True),),

            nn.Sequential(Conv2dTranspose(576, 384, kernel_size=3, stride=1, padding=1), # 3,3
            Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),),

            nn.Sequential(Conv2dTranspose(640, 384, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),), # 6, 6

            nn.Sequential(Conv2dTranspose(640, 384, kernel_size=3, stride=1, padding=1),
            Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),), # 6, 6

            nn.Sequential(Conv2dTranspose(640, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),), # 12, 12

            nn.Sequential(Conv2dTranspose(384, 128, kernel_size=3, stride=1, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),), # 12, 12

            nn.Sequential(Conv2dTranspose(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),), # 24, 24

            nn.Sequential(Conv2dTranspose(256, 128, kernel_size=3, stride=1, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),), # 24, 24

            nn.Sequential(Conv2dTranspose(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1), 
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),), # 48, 48

            nn.Sequential(Conv2dTranspose(192, 96, kernel_size=3, stride=1, padding=1), 
            Conv2d(96, 96, kernel_size=3, stride=1, padding=1, residual=True),), # 48, 48

            nn.Sequential(Conv2dTranspose(160, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),), # 96,96

            nn.Sequential(Conv2dTranspose(128, 64, kernel_size=3, stride=1, padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),), # 96,96
            
            nn.Sequential(
                Conv2dTranspose(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                
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

class Wav2Lip_disc_qual(nn.Module):
    def __init__(self):
        super(Wav2Lip_disc_qual, self).__init__()

        self.face_encoder_blocks = nn.ModuleList([
            # Added by eddy
            nn.Sequential(nonorm_Conv2d(3, 32, kernel_size=7, stride=1, padding=3)), # 96,192
            nn.Sequential(nonorm_Conv2d(3, 32, kernel_size=7, stride=2, padding=3)), # 96,192
            # End added by eddy

            nn.Sequential(nonorm_Conv2d(3, 32, kernel_size=7, stride=1, padding=3)), # 48,96

            nn.Sequential(nonorm_Conv2d(32, 64, kernel_size=5, stride=(1, 2), padding=2), # 48,48
            nonorm_Conv2d(64, 64, kernel_size=5, stride=1, padding=2)),

            nn.Sequential(nonorm_Conv2d(64, 128, kernel_size=5, stride=2, padding=2),    # 24,24
            nonorm_Conv2d(128, 128, kernel_size=5, stride=1, padding=2)),

            nn.Sequential(nonorm_Conv2d(128, 256, kernel_size=5, stride=2, padding=2),   # 12,12
            nonorm_Conv2d(256, 256, kernel_size=5, stride=1, padding=2)),

            nn.Sequential(nonorm_Conv2d(256, 512, kernel_size=3, stride=2, padding=1),       # 6,6
            nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=1)),

            nn.Sequential(nonorm_Conv2d(512, 512, kernel_size=3, stride=2, padding=1),     # 3,3
            nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=1),),
            
            nn.Sequential(nonorm_Conv2d(512, 512, kernel_size=3, stride=1, padding=0),     # 1, 1
            nonorm_Conv2d(512, 512, kernel_size=1, stride=1, padding=0)),])

        self.binary_pred = nn.Sequential(nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0), nn.Sigmoid())
        self.label_noise = .0

    def get_lower_half(self, face_sequences):
        return face_sequences[:, :, face_sequences.size(2)//2:]

    def to_2d(self, face_sequences):
        B = face_sequences.size(0)
        face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)
        return face_sequences

    def perceptual_forward(self, false_face_sequences):
        false_face_sequences = self.to_2d(false_face_sequences)
        false_face_sequences = self.get_lower_half(false_face_sequences)

        false_feats = false_face_sequences
        for f in self.face_encoder_blocks:
            false_feats = f(false_feats)

        false_pred_loss = F.binary_cross_entropy(self.binary_pred(false_feats).view(len(false_feats), -1), 
                                        torch.ones((len(false_feats), 1)).cuda())

        return false_pred_loss

    def forward(self, face_sequences):
        face_sequences = self.to_2d(face_sequences)
        face_sequences = self.get_lower_half(face_sequences)

        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)

        return self.binary_pred(x).view(len(x), -1)
