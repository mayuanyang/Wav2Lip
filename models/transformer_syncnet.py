import torch
from torch import nn
from torch.nn import functional as F
from .conv import Conv2d
from .self_attention import AttentionBlock
from .cross_modal_attention import CrossModalAttention2d

def check_nan(tensor, name):
    if torch.isnan(tensor).any():
        print('NaN problem', f"NaN in {name}")

import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3, num_layers=4):
        super(SpatialAttention, self).__init__()
        layers = []
        padding = (kernel_size - 1) // 2
        # Start with 2-channel input (from avg and max pooling)
        in_channels = 2
        hidden_channels = 16  # Arbitrary choice; adjust as needed
        
        # Add intermediate layers
        for _ in range(num_layers - 1):
            layers.append(nn.Conv2d(in_channels, hidden_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.ReLU(inplace=True))
            in_channels = hidden_channels
        
        # Final convolution to get a single-channel attention map
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=kernel_size, padding=padding, bias=False))
        self.conv = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()
        # Learnable scaling factor to adjust the contribution of the attention
        self.alpha = nn.Parameter(torch.zeros(1))  # Initialized to zero (or a small value)
    
    
    def forward(self, x):
        # x: (B, C, H, W)
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)
        x_cat = torch.cat([avg_pool, max_pool], dim=1)     # (B, 2, H, W)
        attn = self.conv(x_cat)
        attn = self.sigmoid(attn)
        out = x + self.alpha * attn
        return out

class TransformerSyncnet(nn.Module):
    def __init__(self, num_heads, num_encoder_layers):
        super(TransformerSyncnet, self).__init__()
        # --- Face encoder for individual frames ---
        self.face_encoder_individual = nn.Sequential(
            Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True), 
            
            nn.MaxPool2d(2, 2),
            Conv2d(32, 64, kernel_size=3, stride=1, padding=1), 
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), 
            
            nn.MaxPool2d(2, 2),
            Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            
            nn.MaxPool2d(2, 2),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            
            nn.MaxPool2d(2, 2),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            
            Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            
            Conv2d(256, 256, kernel_size=3, stride=(1,2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            
            Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            
            Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            
            Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            
            Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Flatten()
        )
        
        # --- Audio encoder (as in original implementation) ---
        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
                        
            Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            
            Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            
            nn.MaxPool2d(2, 2),
            
            Conv2d(128, 256, kernel_size=3, stride=(2,1), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            
            Conv2d(256, 256, kernel_size=3, stride=(2,1), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            
            Conv2d(256, 256, kernel_size=3, stride=(2,1), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            
            Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            
            Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
        )
        
        # --- Token projection ---
        # Project both face tokens and the audio feature to the transformer's model dimension.
        self.token_proj = nn.Linear(256, 512)
        self.audio_proj = nn.Linear(256, 512)
        
        # --- Learnable CLS token ---
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 512))
        
        # --- Learnable Positional Encoding ---
        # There are 7 tokens: 1 CLS, 5 face tokens, and 1 audio token.
        self.pos_embedding = nn.Parameter(torch.zeros(1, 7, 512))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # --- Transformer Encoder ---
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=num_heads, dropout=0.2),
            num_layers=num_encoder_layers
        )
        
        self.relu = nn.LeakyReLU(0.01, inplace=True)
        # Final classification layer
        self.fc3 = nn.Linear(512, 1)
        
    def forward(self, face_embedding, audio_embedding):
        B, C, H, W = face_embedding.shape  # Expected: C == 15 (i.e., 5 images x 3 channels)
        num_frames = 5
        channels_per_frame = 3
        
        # --- Process Face Modality as Individual Tokens ---
        # Reshape to separate the 5 images: (B, 5, 3, H, W)
        face_frames = face_embedding.view(B, num_frames, channels_per_frame, H, W)
        # Merge batch and frame dimensions: (B*5, 3, H, W)
        face_frames = face_frames.view(B * num_frames, channels_per_frame, H, W)
        # Process each frame individually
        face_features = self.face_encoder_individual(face_frames)  # (B*5, 256)
        # Reshape back to (B, 5, 256)
        face_features = face_features.view(B, num_frames, 256)
        # Project each face token to the transformer's model dimension
        face_tokens = self.token_proj(face_features)  # (B, 5, 512)
        
        # --- Process Audio Modality ---
        a2 = self.audio_encoder(audio_embedding)
        a2 = a2.view(B, -1)  # (B, 256)
        audio_token = self.audio_proj(a2)  # (B, 512)
        audio_token = audio_token.unsqueeze(1)  # (B, 1, 512)
                
        # --- Form the Combined Token Sequence ---
        # Prepend a learnable classification token.
        cls_tokens = self.cls_token.expand(B, 1, 512)  # (B, 1, 512)
        # Concatenate tokens: [CLS] + (5 face tokens) + (1 audio token) => (B, 7, 512)
        combined_tokens = torch.cat([cls_tokens, face_tokens, audio_token], dim=1)
        
        # --- Add Positional Encoding ---
        # The pos_embedding is broadcasted along the batch dimension.
        combined_tokens = combined_tokens + self.pos_embedding
        
        # --- Rearrange for Transformer ---
        # Transformer expects input shape: (sequence_length, batch_size, d_model)
        combined_tokens = combined_tokens.transpose(0, 1)  # (7, B, 512)
        
        # --- Transformer Encoding ---
        transformer_output = self.transformer_encoder(combined_tokens)  # (7, B, 512)
        aggregated_output = transformer_output.mean(dim=0)  # (B, 512)
        
        out = self.relu(aggregated_output)
        out = self.fc3(out)
        
        return out, None, None
