import torch
from torch import nn
from torch.nn import functional as F
import math

from .conv import Conv2dTranspose, Conv2d, nonorm_Conv2d

class Wav2Lip(nn.Module):
    def __init__(self, num_of_blocks=2):
        super(Wav2Lip, self).__init__()
        self.blocks = nn.ModuleList()

        for i in range(num_of_blocks):
            self.blocks.append(ProcessBlock(9))
        
        print('The length of blocks', len(self.blocks))
        self.output_block = ProcessBlock(3)

        # Define the sharpening kernel
        sharpen_kernel = torch.tensor([[[[ 0, -1,  0],
                                         [-1,  5, -1],
                                         [ 0, -1,  0]]]], dtype=torch.float32)
        
        # Sharpening layer as a convolution with fixed weights
        self.sharpen = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False, groups=3)
        self.sharpen.weight = nn.Parameter(sharpen_kernel.repeat(3, 1, 1, 1), requires_grad=False)
                
        
    def forward(self, audio_sequences, face_sequences, sharpen_img):
        temp_output = None
        face_input = face_sequences
        for block in self.blocks:
            if temp_output is not None:
                face_input = face_sequences + temp_output
            
            temp_output = self.forward_impl(audio_sequences, face_input, block.face_encoder_blocks, block.audio_encoder, block.face_decoder_blocks, block.output_block, sharpen_img)
            
        step2_face_sequences = face_sequences + temp_output
        outputs = self.forward_impl(audio_sequences, step2_face_sequences, self.output_block.face_encoder_blocks, self.output_block.audio_encoder, self.output_block.face_decoder_blocks, self.output_block.output_block, sharpen_img)
      
        return outputs

    def forward_impl(self, audio_sequences, face_sequences, face_encoder_blocks, audio_encoder, face_decoder_blocks, output_block, sharpen_img):
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

        x = audio_embedding
        index = 1
        for f in face_decoder_blocks:
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
        x = output_block(x)
        
        if sharpen_img:
          channels = x.shape[1]
          if channels == 3:
                x = self.sharpen(x)

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
