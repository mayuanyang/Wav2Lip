
import torch
from torch import nn
from torch.nn import functional as F
from .conv import Conv2d
from .cross_attention import CrossAttention

def initialize_weights(module):
    """
    Initialize weights for different types of layers.
    
    Args:
        module (nn.Module): The module to initialize.
    """
    if isinstance(module, nn.Conv2d):
        # Initialize Conv2d with Kaiming Normal for ReLU activations
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        # Initialize Linear layers with Kaiming Normal
        nn.init.kaiming_normal_(module.weight, a=0, mode='fan_in', nonlinearity='linear')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm2d):
        # Initialize BatchNorm layers
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        # Initialize LayerNorm layers
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

class TransformerSyncnetV2(nn.Module):
    def __init__(self, num_cross_attn_layers=2, embed_dim=512, num_heads=8, dropout=0.1):
        super(TransformerSyncnetV2, self).__init__()
        '''
        Outpu = Input + (k-1) x S

        Where:
        Input is the receptive field size from the previous layer.
        k is the kernel size.
        S is the stride.
        '''

        self.face_encoder = nn.Sequential(
            
            Conv2d(15, 128, kernel_size=3, stride=1, padding=1), #192x192, 1+(3−1)×1=3
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), #192x192, 3+(3−1)×1=5
            
            Conv2d(128, 256, kernel_size=3, stride=2, padding=1), #96x96, 7
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True), # 9

            Conv2d(256, 256, kernel_size=3, stride=(1, 2), padding=1), #94x47, 13
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True), #94x47, 15

            Conv2d(256, 256, kernel_size=3, stride=2, padding=1), # 47x24, 19
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True), # 47x24, 21

            Conv2d(256, 512, kernel_size=3, stride=2, padding=1), # 24x 12, 25
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), # 24x 12, 27

            Conv2d(512, 512, kernel_size=3, stride=1, padding=1), # 24x 12, 31
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), # 24x 12, 33

            Conv2d(512, 512, kernel_size=3, stride=2, padding=1), #12x6, 39
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), # 24x 12, 27

            Conv2d(512, 512, kernel_size=3, stride=2, padding=1), #12x6, 39
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), # 24x 12, 27

            Conv2d(512, 256, kernel_size=3, stride=2, padding=1), #12x6, 39
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True), # 24x 12, 27

            Conv2d(256, 128, kernel_size=1, stride=1, padding=0), #12x6, 39
            

            )

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1), # 80x16, 1+(3−1)×1=3
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True), # 3+(3-1)x1=5
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True), # 3+(3-1)x1=5

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1), # 27x16, 11+(3-1)x3=17
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), #15+(3-1)x1=19
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), #15+(3-1)x1=19

            Conv2d(64, 64, kernel_size=3, stride=2, padding=1), # 14x8, 25+(3-1)x2=29
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), #29+(3-1)x1=31

            Conv2d(64, 128, kernel_size=3, stride=(2,1), padding=1), #7x8, 35+(3-1)x2=41
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), #41+(3-1)x1=43
            

            Conv2d(128, 256, kernel_size=3, stride=3, padding=1), #3x3, #47+(3-1)x3=51
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True), #51+(3-1)x1=53

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0), #3x3, 53+(3-1)x1=55
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0, residual=True),) #55+(3-1)x1=57

                
        # Separate Cross-Attention Layers for Each Direction
        self.cross_attn_face_to_audio = nn.ModuleList([
            CrossAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_cross_attn_layers)
        ])
        self.cross_attn_audio_to_face = nn.ModuleList([
            CrossAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_cross_attn_layers)
        ])
        
        
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, 2)
        self.dropout2 = nn.Dropout(dropout)

        # Initialize FC layers
        self.fc2.apply(initialize_weights)

        # Initialize entire model
        self.apply(initialize_weights)
        

    def forward(self, face_embedding, audio_embedding):

        # Encode face and audio
        face_embedding = self.face_encoder(face_embedding)    # (batch_size, 1536)
        audio_embedding = self.audio_encoder(audio_embedding) # (batch_size, 1536)

        audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)
        face_embedding = face_embedding.view(face_embedding.size(0), -1)

                
        # Apply Multiple Cross-Attention Layers with Residual Connections
        for i in range(len(self.cross_attn_face_to_audio)):
            # Cross-Attention: Face attends to Audio
            face_to_audio = self.cross_attn_face_to_audio[i](face_embedding, audio_embedding, audio_embedding)  # (batch_size, embed_dim)
            # Cross-Attention: Audio attends to Face
            audio_to_face = self.cross_attn_audio_to_face[i](audio_embedding, face_embedding, face_embedding)  # (batch_size, embed_dim)
            
            # Residual Connections
            face_embedding = face_embedding + face_to_audio  # (batch_size, embed_dim)
            audio_embedding = audio_embedding + audio_to_face  # (batch_size, embed_dim)
        
        # Combine the final embeddings
        combined = face_embedding + audio_embedding  # (batch_size, embed_dim)

        # Classification Layers
        combined = self.dropout1(combined)
        out = self.fc2(combined)                   # (batch_size, 2)
        out = self.dropout2(out)
        
        return out, audio_embedding, face_embedding
