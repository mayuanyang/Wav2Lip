import torch
from torch import nn
from torch.nn import functional as F
import math

from .conv import Conv2dTranspose, Conv2d, nonorm_Conv2d

class TransformerWav2Lip(nn.Module):
    def __init__(self):
        super(TransformerWav2Lip, self).__init__()

        self.face_encoder = nn.Sequential(
            
            Conv2d(6, 32, kernel_size=7, stride=1, padding=3), #192x192, 1+(7−1)×1=7
            
            Conv2d(32, 64, kernel_size=7, stride=2, padding=3), #96x96, 7+(7−1)×2=7+12=19
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), # 19+(3−1)×1=19+2=21
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), # 21+(3−1)×1=19+2=23

            Conv2d(64, 128, kernel_size=5, stride=(1, 2), padding=1), #94x47, 23+(5−1)×1=19+2=23
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), #94x47
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), #94x47

            Conv2d(128, 256, kernel_size=5, stride=2, padding=2), # 47x24
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True), # 47x24
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True), # 47x24

            Conv2d(256, 256, kernel_size=5, stride=2, padding=2), # 24x 12
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True), # 24x 12
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True), # 24x 12

            Conv2d(256, 512, kernel_size=3, stride=1, padding=1), # 24x 12
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), # 24x 12
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), # 24x 12

            Conv2d(512, 256, kernel_size=5, stride=2, padding=2), #12x6
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True), #12x6
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True), #12x6

            Conv2d(256, 128, kernel_size=3, stride=(2,1), padding=1), #6x6
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), #12x6
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), #12x6

            Conv2d(128, 64, kernel_size=3, stride=1, padding=1), #6x6
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1), #6x6

            Conv2d(64, 64, kernel_size=3, stride=2, padding=0), #2x2 The receptive field up to here is 99x99
            
            )

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1), # 80x16
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 128, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            Conv2d(128, 128, kernel_size=1, stride=1, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=2, padding=0),)

        
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=768, nhead=8, dropout=0.1),
            num_layers=2
        )

        self.upsample_block = nn.Sequential(
            # First upsampling from 4x4 to 8x8
            nn.ConvTranspose2d(48, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Upsample from 8x8 to 16x16
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Upsample from 16x16 to 32x32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Upsample from 32x32 to 64x64
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            # Upsample from 64x64 to 128x128
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            
            # Upsample from 128x128 to 192x192
            nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            
            # Final Conv layer to match the desired number of output channels
            nn.Conv2d(3, 3, kernel_size=3, padding=1),
            nn.Sigmoid()  # Sigmoid to bring the values between 0 and 1 for an image
        )
        
        # self.fc1 = nn.Linear(3538944, 512) 
        
        # self.transformer_encoder = nn.TransformerEncoderLayer(d_model=512, nhead=8)  # Adjust d_model and nhead as per your need

        # self.output_block = nn.Sequential(Conv2d(48, 32, kernel_size=3, stride=1, padding=1),
        #     nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
        #     nn.Sigmoid())
        
        

    def forward(self, audio_sequences, face_sequences):
        '''
        The face sequence has the shape of (B, 6, 5, 192, 192)
        where B is the batch size 
        6 is the channels, it is the concatenation of window and wrong window's channel, each one has 3 channels
        5 is the number of images
        192x192 is the H and W
        '''
        # print('The audio input shape', audio_sequences.shape)
        # print('The face input shape', face_sequences.shape)

        # audio_sequences = (B, T, 1, 80, 16)
        B = audio_sequences.size(0)

        input_dim_size = len(face_sequences.size())
        if input_dim_size > 4:
            '''
            For audio, concat the dim 1 which is the time step
            For face, concat the dim 2 which is the number of images(5)
            '''
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)

        # print('The audio shape before encoder', audio_sequences.shape)
        # print('The face shape before encoder', face_sequences.shape)

        face_embedding = self.face_encoder(face_sequences)
        audio_embedding = self.audio_encoder(audio_sequences)

        # print('The face shape after encoder', face_embedding.shape)
        # print('The audio shape after encoder', audio_embedding.shape)


        '''
        Keep the first 2 dimension and flatten the rest dimension into 1, flatten means multiply the rest dimension
        So the shape after the permute(change order) become [Batch, image_size(HxW), Channel]
        '''
        audio_embedding = audio_embedding.view(audio_embedding.size(0), audio_embedding.size(1), -1)
        face_embedding = face_embedding.view(face_embedding.size(0), face_embedding.size(1), -1)
        
        audio_embedding = audio_embedding.permute(0, 2, 1)
        face_embedding = face_embedding.permute(0, 2, 1)

        # print('The face shape', face_embedding.shape)
        # print('The audio shape', audio_embedding.shape)

        
        # normalise them
        audio_embedding = F.normalize(audio_embedding, p=2, dim=2)
        face_embedding = F.normalize(face_embedding, p=2, dim=2)

        # Concatenate lip frames and audio features
        combined = torch.cat((face_embedding, audio_embedding), dim=2)

        # print('The combined shape', combined.shape)
        
        # Make sure combined is 1-dimensional
        combined = combined.view(combined.size(0), -1)

        # Pass through the Transformer encoder, the input size is 1024
        transformer_output = self.transformer_encoder(combined)

        # print('The transformer output shape', transformer_output.shape)

        # Step 2: Reshape it back to the original shape [batch_size, channels, height, width]
        batch_size = combined.shape[0]
        
        transformer_output = transformer_output.view(batch_size, 48, 4, 4) 

        # print('The unflatten transformer output shape', transformer_output.shape)

        x = self.upsample_block(transformer_output)

        if input_dim_size > 4:
            x = torch.split(x, B, dim=0) # [(B, C, H, W)]
            outputs = torch.stack(x, dim=2) # (B, C, T, H, W)

        else:
            outputs = x
            
        '''
        The target outputs shape is torch.Size([B, 3, 5, 192, 192])
        '''
        return outputs
