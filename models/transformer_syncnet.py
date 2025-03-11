import torch
from torch import nn, resize_as_
from torch.nn import functional as F
from .conv import Conv2d
from .self_attention import AttentionBlock
from .cross_modal_attention import CrossModalAttention2d
import torch.nn.init as init

def check_nan(tensor, name):
    if torch.isnan(tensor).any():
        print('NaN problem', f"NaN in {name}")

def initialize_weights(module):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=5, num_layers=2):
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
    def __init__(self, num_heads=8, num_encoder_layers=4, embed_dim=512):
        super(TransformerSyncnet, self).__init__()
        
        self.embed_dim = embed_dim
        # --- Face encoder for individual frames ---
        self.face_encoder1 = nn.Sequential(
            # Input: (B, 15, H, W)  where 15 = 5 images x 3 channels
            Conv2d(15, 64, kernel_size=3, stride=1, padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), 
            
            Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Downsample
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), 
        )
        
        self.face_encoder2 = nn.Sequential(    
            Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # Downsample
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            
            Conv2d(256, 512, kernel_size=3, stride=(1,2), padding=1),  # Downsample width
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
        )
        
        self.face_encoder3 = nn.Sequential(
            Conv2d(512, 512, kernel_size=3, stride=2, padding=1),  # Downsample
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
        )
        
        self.face_skip1 = nn.Sequential(    
            Conv2d(15, 128, kernel_size=3, stride=2, padding=1),  # Downsample
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
        )
        
        self.face_skip2 = nn.Sequential(    
            Conv2d(15, 512, kernel_size=3, stride=2, padding=1),  # Downsample
            Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=(1,2), padding=1),
        )
        
        # --- Audio encoder ---
        self.audio_encoder = nn.Sequential(
            # Example input shape: (B, 1, H_audio, W_audio)
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            
            Conv2d(32, 64, kernel_size=3, stride=(2,1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            
            Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            
            Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            
            Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
        )
        
        # Projection layers: ensure both modalities have the same embedding dimension.
        # Here, both face and audio encoders output 512 channels.
        self.face_proj = nn.Conv2d(512, embed_dim, kernel_size=1)
        self.audio_proj = nn.Conv2d(512, embed_dim, kernel_size=1)

        self.face_layer_norm = nn.LayerNorm(embed_dim)
        self.audio_layer_norm = nn.LayerNorm(embed_dim)
        
        # Cross-modal Transformer encoder.
        # We will first flatten the spatial dimensions into a token sequence.
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.cross_modal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # Final classification head.
        # We pool tokens for each modality separately, then concatenate their global features.
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.LeakyReLU(0.01, inplace=False),
            nn.Linear(128, 1)  # binary classification output
        )
        
    
    def forward(self, face_embedding, audio_embedding):
        """
        face_embedding: tensor of shape (B, 15, H, W) -> 5 images concatenated (each 3 channels)
        audio_embedding: tensor of shape (B, 1, H_audio, W_audio)
        """
        # --- Process face modality ---
        
        face1 = self.face_encoder1(face_embedding)
        face_skip1 = self.face_skip1(face_embedding)
        face1 = face1 + face_skip1
        
        face2 = self.face_encoder2(face1)
        face_skip2 = self.face_skip2(face_embedding)
        face2 = face2 + face_skip2
        
        face3 = self.face_encoder3(face2)
        
        face_features = face3  # (B, 512, H_f, W_f)
        
        # --- Process audio modality ---
        audio_features = self.audio_encoder(audio_embedding)  # (B, 512, H_a, W_a)
        # Interpolate audio features to match face feature spatial dimensions:
        B, _, H, W = face_features.shape
        audio_features = F.interpolate(audio_features, size=(H, W), mode='bilinear', align_corners=False)
        
        # --- Project both modalities to the common embedding dimension ---
        face_proj = self.face_proj(face_features)   # (B, embed_dim, H, W)
        audio_proj = self.audio_proj(audio_features)  # (B, embed_dim, H, W)
        
        # --- Flatten spatial dimensions into tokens ---
        # Resulting shape: (num_tokens, B, embed_dim) where num_tokens = H * W.
        face_tokens = face_proj.view(B, -1, H * W).permute(2, 0, 1)
        audio_tokens = audio_proj.view(B, -1, H * W).permute(2, 0, 1)

        face_tokens = self.face_layer_norm(face_tokens)
        audio_tokens = self.audio_layer_norm(audio_tokens)
        
        # --- Concatenate the token sequences along the token dimension ---
        # Combined tokens shape: (2 * num_tokens, B, embed_dim)
        combined_tokens = torch.cat([face_tokens, audio_tokens], dim=0)
        
        # --- Apply cross-modal Transformer encoder ---
        combined_tokens = self.cross_modal_transformer(combined_tokens)

        
        # --- Separate tokens back by modality ---
        num_tokens = H * W
        face_tokens_out = combined_tokens[:num_tokens, :, :]  # (num_tokens, B, embed_dim)
        audio_tokens_out = combined_tokens[num_tokens:, :, :]   # (num_tokens, B, embed_dim)
        
        # --- Global average pooling for each modality ---
        face_global = face_tokens_out.max(dim=0)[0]   # (B, embed_dim)
        audio_global = audio_tokens_out.max(dim=0)[0] # (B, embed_dim)
        
        # --- Fuse modalities ---
        fused_features = torch.cat([face_global, audio_global], dim=1)  # (B, 2*embed_dim)
        
        # --- Classification head ---
        logits = self.classifier(fused_features)  # (B, 1)
        
        return logits, None, None
