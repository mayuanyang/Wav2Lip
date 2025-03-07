import torch
from torch import nn
from torch.nn import functional as F
from .conv import Conv2d
from .self_attention import AttentionBlock
from .cross_modal_attention import CrossModalAttention2d

def check_nan(tensor, name):
    if torch.isnan(tensor).any():
        print('NaN problem', f"NaN in {name}")

class TransformerSyncnet(nn.Module):
    def __init__(self, num_heads, num_encoder_layers):
        super(TransformerSyncnet, self).__init__()
        # --- Face encoder for individual frames ---
        # This encoder processes a single 3-channel image and outputs a 512-d feature.
        self.face_encoder_individual = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
            nn.Flatten(),                   # Shape becomes (B, 128)
            nn.Linear(128, 512)             # Project to 512-dim feature
        )
        
        # --- Audio encoder (as in original implementation) ---
        self.audio_encoder1 = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            
            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            
            Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
        )
        
        self.audio_encoder2 = nn.Sequential(
            Conv2d(128, 128, kernel_size=3, stride=(2, 1), padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            
            Conv2d(128, 256, kernel_size=3, stride=3, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            
            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0, residual=True),
        )
        
        # --- Token projection ---
        # Both face tokens (from individual frames) and the aggregated audio feature will be 512-d,
        # so we project them to the transformer’s model dimension (d_model=1024).
        self.token_proj = nn.Linear(512, 1024)
        
        # --- Learnable CLS token ---
        # This token will be used for the final classification.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 1024))
        
        # --- Transformer Encoder ---
        # Note: the transformer expects input shape [seq_len, batch, d_model]
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=1024, nhead=num_heads, dropout=0.2),
            num_layers=num_encoder_layers
        )
        
        self.relu = nn.LeakyReLU(0.01, inplace=True)
        # Final classification layer; output dimension (e.g., 2 classes)
        self.fc3 = nn.Linear(1024, 2)
        
    def forward(self, face_embedding, audio_embedding):
        B, C, H, W = face_embedding.shape  # Expected: C == 15 (i.e. 5 images x 3 channels)
        num_frames = 5
        channels_per_frame = 3
        
        # --- Process Face Modality as Individual Tokens ---
        # Reshape to separate the 5 images: (B, 5, 3, H, W)
        face_frames = face_embedding.view(B, num_frames, channels_per_frame, H, W)
        # Merge batch and frame dimensions to process all frames in parallel: (B*5, 3, H, W)
        face_frames = face_frames.view(B * num_frames, channels_per_frame, H, W)
        # Process each frame individually
        face_features = self.face_encoder_individual(face_frames)  # Shape: (B*5, 512)
        # Reshape back to (B, 5, 512)
        face_features = face_features.view(B, num_frames, 512)
        # Normalize each token along the feature dimension
        face_features = F.normalize(face_features, p=2, dim=2)
        # Project each face token to the transformer's model dimension (1024)
        face_tokens = self.token_proj(face_features)  # Shape: (B, 5, 1024)
        
        # --- Process Audio Modality ---
        audio_embedding1 = self.audio_encoder1(audio_embedding)
        audio_embedding2 = self.audio_encoder2(audio_embedding1)
        # Flatten the audio feature map to get a vector per sample.
        audio_embedding2 = audio_embedding2.view(B, -1)
        audio_embedding2 = F.normalize(audio_embedding2, p=2, dim=1)
        # Project the aggregated audio feature to 1024.
        audio_token = self.token_proj(audio_embedding2)  # Shape: (B, 1024)
        # Add a time dimension (token dimension): (B, 1, 1024)
        audio_token = audio_token.unsqueeze(1)
        
        # --- Form the Combined Token Sequence ---
        # Prepend a learnable classification token.
        cls_tokens = self.cls_token.expand(B, 1, 1024)  # (B, 1, 1024)
        # Concatenate tokens: [CLS] + (5 face tokens) + (1 audio token) => (B, 7, 1024)
        combined_tokens = torch.cat([cls_tokens, face_tokens, audio_token], dim=1)
        # Rearrange to match the transformer’s expected input shape: (seq_len, B, d_model)
        combined_tokens = combined_tokens.transpose(0, 1)  # (7, B, 1024)
        
        # --- Transformer Encoding ---
        transformer_output = self.transformer_encoder(combined_tokens)  # (7, B, 1024)
        # Use the output corresponding to the CLS token for classification.
        cls_output = transformer_output[0]  # (B, 1024)
        out = self.relu(cls_output)
        out = self.fc3(out)
        
        return out, None, None
