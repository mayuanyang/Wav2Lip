
import torch
from torch import nn
from torch.nn import functional as F
from .conv import Conv2d
from .cross_modal_attention import CrossModalAttention2d
from .self_attention import AttentionBlock

class TransformerSyncnetV2(nn.Module):
    def __init__(self, num_cross_attn_layers=4, embed_dim=512, num_heads=8, dropout=0.1):
        super(TransformerSyncnetV2, self).__init__()
        '''
        Outpu = Input + (k-1) x S

        Where:
        Input is the receptive field size from the previous layer.
        k is the kernel size.
        S is the stride.
        '''
        self.face_encoder1 = nn.Sequential(
            
            Conv2d(15, 128, kernel_size=3, stride=1, padding=1), #192x384
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), 
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
        )
        
        self.face1_attn = AttentionBlock(128, sparse_attention=True)
        
        self.face_encoder2 = nn.Sequential(
            Conv2d(128, 256, kernel_size=3, stride=2, padding=1), #96x192, 7
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True), # 9
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True), # 11
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True), # 11
        )
        
        self.face2_attn = AttentionBlock(256, sparse_attention=True)
        self.face2xaudio = CrossModalAttention2d(256)
        
        self.face_encoder3 = nn.Sequential(

            Conv2d(256, 512, kernel_size=3, stride=(1, 2), padding=1), #96x96
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), 
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), 

            Conv2d(512, 512, kernel_size=3, stride=2, padding=1), # 48x48
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), 
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), 
        )
        
        self.face_encoder4 = nn.Sequential(

            Conv2d(512, 512, kernel_size=3, stride=2, padding=1), # 24x 24
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            
            Conv2d(512, 512, kernel_size=3, stride=2, padding=1), # 12x 12, 31
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            
        )
        
        self.face4xaudio = CrossModalAttention2d(512)

        self.audio_encoder1 = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1), # 80x16
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True), 
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True), 
            

            Conv2d(32, 64, kernel_size=3, stride=(2, 1), padding=1), # 40x16
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), 
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), 
            

            Conv2d(64, 64, kernel_size=3, stride=(2, 1), padding=1), # 20x16
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), 
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), 
            

            Conv2d(64, 128, kernel_size=3, stride=2, padding=1), #10x8
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            
            Conv2d(128, 256, kernel_size=3, stride=1, padding=(1,2)), #10x10, 
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True), 
        )
        
        self.audio_encoder2 = nn.Sequential(
            Conv2d(256, 512, kernel_size=3, stride=1, padding=2), #12x12
        )
        
        self.flatten = Conv2d(512, 512, kernel_size=1, stride=1, padding=1)
        
        self.fc = nn.Linear(512, 2) 
        

    def forward(self, face_embedding, audio_embedding):
      
        audio_embedding1 = self.audio_encoder1(audio_embedding)
        audio_embedding2 = self.audio_encoder2(audio_embedding1)
        
        print('The shapes', audio_embedding.shape)
        
        # Encode the face and audio embeddings
        face1 = self.face_encoder1(face_embedding)
        face1 = self.face1_attn(face1)
        
        face2 = self.face_encoder2(face1)
        face2 = self.face2_attn(face2)
        face2 = self.face2xaudio(face2, audio_embedding1)
        
        face3 = self.face_encoder3(face2)
        face4 = self.face_encoder4(face3)
        face4 = self.face4xaudio(face4, audio_embedding2)
        
        flatten = self.flatten(face4)

        out = self.fc3(flatten)

        return out, audio_embedding, face_embedding

