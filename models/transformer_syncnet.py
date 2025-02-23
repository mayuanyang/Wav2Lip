
import torch
from torch import nn
from torch.nn import functional as F
from .conv import Conv2d

class TransformerSyncnet(nn.Module):
    def __init__(self, num_heads, num_encoder_layers):
        super(TransformerSyncnet, self).__init__()
        '''
        Outpu = Input + (k-1) x S

        Where:
        Input is the receptive field size from the previous layer.
        k is the kernel size.
        S is the stride.
        '''
        self.face_encoder = nn.Sequential(
            
            Conv2d(15, 64, kernel_size=3, stride=1, padding=1), #192x192, 1+(3−1)×1=3
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), #192x192, 3+(3−1)×1=5
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), #192x192, 3+(3−1)×1=5
            
            Conv2d(64, 64, kernel_size=3, stride=2, padding=1), #96x96, 7
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), # 9
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), # 11

            Conv2d(64, 128, kernel_size=3, stride=(1, 2), padding=1), #94x47, 13
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), #94x47, 15
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), #94x47, 17

            Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # 47x24, 19
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True), # 47x24, 21
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True), # 47x24, 23

            Conv2d(256, 256, kernel_size=3, stride=2, padding=1), # 24x 12, 25
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True), # 24x 12, 27
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True), # 24x 12, 29

            Conv2d(256, 512, kernel_size=3, stride=1, padding=1), # 24x 12, 31
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), # 24x 12, 33
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), # 24x 12, 35

            Conv2d(512, 256, kernel_size=3, stride=2, padding=1), #12x6, 37
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True), #12x6, 39
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True), #12x6, 41

            Conv2d(256, 256, kernel_size=3, stride=(2,1), padding=1), #6x6, 43
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True), # 45
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True), # 47

            Conv2d(256, 128, kernel_size=3, stride=1, padding=1), #6x6, 49
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), #6x6, 51

            Conv2d(128, 128, kernel_size=3, stride=1, padding=1), #6x6, 53
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), #3x3, 55

            Conv2d(128, 64, kernel_size=3, stride=2, padding=1), #3x3, 57
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), #3x3, 59
            
            )

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1), # 80x16, 1+(3−1)×1=3
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True), # 3+(3-1)x1=5
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True), # 5+(3-1)x1=7
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True), # 7+(3-1)x1=9
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True), # 7+(3-1)x1=11

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1), # 27x16, 11+(3-1)x3=17
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), #15+(3-1)x1=19
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), #17+(3-1)x1=21
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), #19+(3-1)x1=23
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), #19+(3-1)x1=25

            Conv2d(64, 64, kernel_size=3, stride=2, padding=1), # 14x8, 25+(3-1)x2=29
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), #29+(3-1)x1=31
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), #29+(3-1)x1=33
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), #29+(3-1)x1=35

            Conv2d(64, 128, kernel_size=3, stride=(2,1), padding=1), #7x8, 35+(3-1)x2=41
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), #41+(3-1)x1=43
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), #43+(3-1)x1=45
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), #45+(3-1)x1=47

            Conv2d(128, 256, kernel_size=3, stride=3, padding=1), #3x3, #47+(3-1)x3=51
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True), #51+(3-1)x1=53

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0), #3x3, 53+(3-1)x1=55
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0, residual=True),) #55+(3-1)x1=57


        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=num_heads, dropout=0.1),
            num_layers=num_encoder_layers
        )

        self.relu = nn.LeakyReLU(0.01, inplace=True)
        self.fc1 = nn.Linear(384, 256) 
        self.fc2 = nn.Linear(512, 256) 
        self.fc3 = nn.Linear(512, 2) 
        

    def forward(self, face_embedding, audio_embedding):

        face_embedding = self.face_encoder(face_embedding)
        audio_embedding = self.audio_encoder(audio_embedding)

        

        audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)
        face_embedding = face_embedding.view(face_embedding.size(0), -1)

        face_embedding = self.fc1(face_embedding)
        audio_embedding = self.fc2(audio_embedding)

        # normalise them
        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        face_embedding = F.normalize(face_embedding, p=2, dim=1)

        # Concatenate lip frames and audio features
        combined = torch.cat((face_embedding, audio_embedding), dim=1)
        
        # Make sure combined is 1-dimensional
        combined = combined.view(combined.size(0), -1)

        # Pass through the Transformer encoder, the input size is 1024
        transformer_output = self.transformer_encoder(combined)
        out = self.relu(transformer_output)
        out = self.fc3(out)
        
        return out, audio_embedding, face_embedding