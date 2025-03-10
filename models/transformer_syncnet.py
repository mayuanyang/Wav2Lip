import torch
from torch import nn
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
        self.face_encoder_individual = nn.Sequential( # 192x384
            Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True), 
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True), 
            
            Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 96x192
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), 
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), 
            
            
            Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 48x96
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            
            
            Conv2d(128, 64, kernel_size=3, stride=(1,2), padding=1), # 48x48
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            
            Conv2d(64, 32, kernel_size=3, stride=1, padding=1), 
        )
        
        # --- Audio encoder (as in original implementation) ---
        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
                        
            Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            
            Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            
            Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            
            Conv2d(64, 32, kernel_size=3, stride=1, padding=1), 
        )
        
        
        # --- Token projection ---
        # Project both face tokens and the audio feature to the transformer's model dimension.
        self.face_tokens_proj = nn.Linear(8192, 512)
        self.audio_tokens_proj = nn.Linear(8192, 512)
        
        # --- Learnable CLS token ---
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 512))
        
        # --- Learnable Positional Encoding ---
        # There are 7 tokens: 1 CLS, 5 face tokens, and 1 audio token.
        self.pos_embedding = nn.Parameter(torch.zeros(1, 51, 512))
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
        
        self.apply(initialize_weights)
        
                
    def split_into_16x16_patches(self, tensor, patch_size=16):
        """
        Splits an input tensor of shape (B, C, H, W) into non-overlapping patches of size (patch_size, patch_size).

        Args:
            tensor (torch.Tensor): Input tensor of shape (B, C, H, W). For this example, shape = (29, 256, 80, 16)
            patch_size (int): Size of the patch along both height and width. Default is 16.

        Returns:
            torch.Tensor: A tensor of patches with shape (B, num_patches, C, patch_size, patch_size),
                          where num_patches = (H // patch_size) * (W // patch_size).
                          In this example, num_patches = 5 * 1 = 5.
        """
        B, C, H, W = tensor.shape
        assert H % patch_size == 0, "The height must be divisible by the patch size."
        assert W % patch_size == 0, "The width must be divisible by the patch size."
        
        # Unfold along the height and width dimensions
        patches = tensor.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        # patches shape becomes (B, C, n_patches_h, n_patches_w, patch_size, patch_size)
        n_patches_h = patches.shape[2]
        n_patches_w = patches.shape[3]
        
        # Flatten the patch dimensions into one dimension (n_patches = n_patches_h * n_patches_w)
        patches = patches.contiguous().view(B, C, n_patches_h * n_patches_w, patch_size, patch_size)
        
        # Rearrange to shape: (B, num_patches, C, patch_size, patch_size)
        patches = patches.permute(0, 2, 1, 3, 4)
        return patches
        
    def forward(self, face_embedding, audio_embedding):
        B, C, H, W = face_embedding.shape  # Expected: C == 15 (i.e., 5 images x 3 channels)
        
        num_frames = 5
        channels_per_frame = 3
        
        face_output_channels = 32
        
        # --- Process Face Modality as Individual Tokens ---
        # Reshape to separate the 5 images: (B, 5, 3, H, W)
        face_frames = face_embedding.view(B, num_frames, channels_per_frame, H, W)
        # Merge batch and frame dimensions: (B*5, 3, H, W)
        face_frames = face_frames.view(B * num_frames, channels_per_frame, H, W)
        # Process each frame individually
        face_features = self.face_encoder_individual(face_frames)  # (B*5, 256)
        
        # Extract patches of size (16x16)
        face_patches = self.split_into_16x16_patches(face_features) 
        
        num_of_face_tokens= face_patches.size(1)

        # --- Process Audio Modality ---
        a2 = self.audio_encoder(audio_embedding)
        audio_tokens = self.split_into_16x16_patches(a2)
        
        
        audio_tokens = audio_tokens.reshape(B, 5, -1)        
        
        face_patches = face_patches.view(B, num_frames, num_of_face_tokens, face_output_channels, 16, 16)
        # Merge the 5 groups and 9 tokens into a single token dimension: [29, 45, 256, 16, 16]
        face_tokens = face_patches.reshape(B, num_frames * num_of_face_tokens, -1)
                
        face_tokens = self.face_tokens_proj(face_tokens)
        audio_tokens = self.audio_tokens_proj(audio_tokens)
                
        # --- Form the Combined Token Sequence ---
        # Prepend a learnable classification token.
        cls_tokens = self.cls_token.expand(B, 1, 512)  # (B, 1, 512)
        # Concatenate tokens: [CLS] + (5 face tokens) + (1 audio token) => (B, 7, 512)
        combined_tokens = torch.cat([cls_tokens, face_tokens, audio_tokens], dim=1)
        
                
        # --- Add Positional Encoding ---
        # The pos_embedding is broadcasted along the batch dimension.
        combined_tokens = combined_tokens + self.pos_embedding
        
        # --- Rearrange for Transformer ---
        # Transformer expects input shape: (sequence_length, batch_size, d_model)
        combined_tokens = combined_tokens.transpose(0, 1)  # (7, B, 512)
        
        # --- Transformer Encoding ---
        transformer_output = self.transformer_encoder(combined_tokens)  # (7, B, 512)
        
        aggregated_output = transformer_output[0]
                        
        out = self.relu(aggregated_output)
        out = self.fc3(out)
        
        return out, None, None
