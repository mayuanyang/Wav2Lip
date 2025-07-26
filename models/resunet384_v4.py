import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import GaussianBlur
from .transformer_syncnet import LearnablePositionalEncoding2D
from .spatial_attention import SpatialAttention
from .conv import Conv2dTranspose, Conv2d
import math
import os, cv2
import numpy as np



class CrossAttentionBlock(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, num_heads, dropout=0.1):
        super().__init__()
        assert query_dim % num_heads == 0, "query_dim must be divisible by num_heads"
        
        half_query = query_dim // 2
        self.to_q = nn.Linear(query_dim, half_query)
        self.to_k = nn.Linear(key_dim, half_query)
        self.to_v = nn.Linear(value_dim, half_query)

        self.mha = nn.MultiheadAttention(embed_dim=half_query, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)
        self.ffn = nn.Sequential(
            nn.Linear(half_query, query_dim * 4),
            nn.GELU(),
            nn.Linear(query_dim * 4, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, query_features, key_value_features):
        B_q, C_q, H_q, W_q = query_features.shape
        B_kv, C_kv, H_kv, W_kv = key_value_features.shape

        # Flatten spatial dimensions to sequence length
        query = query_features.view(B_q, C_q, -1).permute(0, 2, 1) # (B, H_q*W_q, C_q)
        key_value = key_value_features.view(B_kv, C_kv, -1).permute(0, 2, 1) # (B, H_kv*W_kv, C_kv)

        # Split query into top and bottom halves
        split_idx = H_q // 2
        top_half = query[:, :split_idx * W_q]
        bottom_half = query[:, split_idx * W_q:]

        # Apply linear projections for Q, K, V
        q = self.to_q(bottom_half)
        k = self.to_k(key_value)
        v = self.to_v(key_value)

        # Cross-attention
        attn_output, _ = self.mha(query=q, key=k, value=v)

        # Add and Norm
        attn_output = self.norm1(attn_output + q) # Residual connection

        # Feed-forward network
        output = self.ffn(attn_output)
        output = self.norm2(output + attn_output) # Residual connection

        # Combine the processed bottom half with the unchanged top half
        output = torch.cat([top_half, output], dim=1)

        # Reshape back to image format (B, C, H, W)
        output = output.permute(0, 2, 1).view(B_q, C_q, H_q, W_q)
        return output

class SparseSelfAttentionBlock(nn.Module):
    
    # MultiheadAttention for SparseSelfAttentionBlock (will be used internally)
    # This is a bit redundant with the above, but illustrates the pattern.
    # In a real implementation, you'd just use a custom attention function or library.
    def __init__(self, in_channels, num_heads, window_size, dropout=0.1):
        super().__init__()
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.window_size = window_size # (window_h, window_w) for local attention

        self.mha = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2),
            nn.GELU(),
            nn.Linear(in_channels * 2, in_channels),
            nn.Dropout(dropout)
        )
        self.qkv_proj = nn.Linear(in_channels, in_channels * 3) # Combined QKV projection
        self.out_proj = nn.Linear(in_channels, in_channels) # Output projection
        self.dropout_layer = nn.Dropout(dropout)

    def generate_dilated_attention_mask(self, H, W, window_size, dilation, device, mask_value=float('-inf')):
        """
        Generates a 2D dilated attention mask for a flattened feature map.

        Args:
            H (int): Height of the feature map.
            W (int): Width of the feature map.
            window_size (tuple): (win_h, win_w) for the local attention window.
                                win_h and win_w should be odd integers.
            dilation (int): The dilation rate for sparse attention.
            device (torch.device): The device to create the mask on.
            mask_value (float or bool): The value to use for masked (disallowed) connections.
                                        -float('inf') for additive attention (e.g., softmax(scores + mask))
                                        -True for boolean mask (e.g., MultiheadAttention(attn_mask=True))

        Returns:
            torch.Tensor: A (H*W) x (H*W) attention mask.
                          If mask_value is -inf, 0 allows attention.
                          If mask_value is True, False allows attention.
        """
        
        N = H * W
        mask = torch.full((N, N), mask_value, device=device, dtype=torch.float32) # Or torch.bool if mask_value is True/False

        half_win_h = window_size[0] // 2
        half_win_w = window_size[1] // 2

        for q_idx in range(N):
            qh, qw = divmod(q_idx, W) # Query (row, col)

            # 1. Local Window Attention
            for kh in range(max(0, qh - half_win_h), min(H, qh + half_win_h + 1)):
                for kw in range(max(0, qw - half_win_w), min(W, qw + half_win_w + 1)):
                    k_idx = kh * W + kw
                    mask[q_idx, k_idx] = 0.0 # Allow attention

            # 2. Dilated Attention (Example: simple row and column dilation)
            # Dilated rows
            for k_offset in range(-H + 1, H): # Iterate over possible row offsets
                kh = qh + k_offset
                if 0 <= kh < H and abs(k_offset) % dilation == 0 and k_offset != 0: # Ensure within bounds and on dilated step
                    kw = qw # Same column
                    k_idx = kh * W + kw
                    mask[q_idx, k_idx] = 0.0

            # Dilated columns
            for k_offset in range(-W + 1, W): # Iterate over possible col offsets
                kw = qw + k_offset
                if 0 <= kw < W and abs(k_offset) % dilation == 0 and k_offset != 0: # Ensure within bounds and on dilated step
                    kh = qh # Same row
                    k_idx = kh * W + kw
                    mask[q_idx, k_idx] = 0.0

        return mask
  
    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        
        # Flatten and permute for attention (B, H*W, C)
        x_flat = x.view(B, C, -1).permute(0, 2, 1) # (B, N, C) where N = H*W
        
        # Residual connection
        residual = x_flat
        
        # Pre-LN Transformer style
        x_norm = self.norm1(x_flat)

        # Apply QKV projection
        qkv = self.qkv_proj(x_norm).chunk(3, dim=-1)
        # q, k, v are now (B, N, C_features) suitable for MHA
        q, k, v = qkv[0], qkv[1], qkv[2]

        # For sparse self-attention, you would generate a sparse attention mask here
        # based on `self.window_size` and pass it to `self.mha`'s `attn_mask` argument.
        # This is where the core logic for *how* it becomes sparse lies.
        # For example, for a local windowed attention, you'd compute Q, K, V for each window
        # and apply attention separately within windows, then stitch them back.
        # This is non-trivial with standard MultiheadAttention without custom kernels.
        
        # As a placeholder, we'll use regular MHA. To truly make it sparse,
        # you'd need a custom attention function or leverage libraries
        # that support sparse attention masks or windowed attention.
        # For instance, if you want local attention, you would process patches.
        # If you want strided/dilated attention, you'd construct a specific mask.
        
        # If you meant a simpler "local" self-attention, the best way often involves
        # using `unfold` and `fold` or processing patches, or specialized layers.
        
        # For now, let's assume a simplified self-attention (not strictly sparse
        # in the standard library's MHA without an explicit mask).
        # We'll pass a dummy mask or no mask, assuming a sparse mechanism
        # would inject its logic here.
        
        # Placeholder for sparse_mask (needs actual implementation based on desired sparsity)
        #sparse_mask = self.generate_dilated_attention_mask(H, W, self.window_size, 2, x.device) # Replace with actual sparse mask generation if needed
                           # e.g., for local attention within a window_size, this mask would be block-diagonal.

        sparse_mask = None
        attn_output, _ = self.mha(query=q, key=k, value=v, attn_mask=sparse_mask)
        
        output = self.dropout_layer(self.out_proj(attn_output))
        
        # Add residual connection
        output = self.norm2(output + residual)
        
        # FFN
        output = self.ffn(output) + output # Another residual connection

        # Reshape back to (B, C, H, W)
        output = output.permute(0, 2, 1).view(B, C, H, W)
        return output

def linear_schedule():
    return torch.tensor([0.3, 0.7, 0.8, 0.9])

def construct_encoder_layers(num_of_layers, input_channels, output_channels, first_layer_stride, add_spatial=False, kernel=3):
    layers = []
    padding = 1
    if kernel == 7:
      padding = 3
    # First layer
    layers.append(Conv2d(input_channels, output_channels, kernel_size=kernel, stride=first_layer_stride, padding=padding))
    # Subsequent layers
    for _ in range(num_of_layers - 1):
        layers.append(Conv2d(output_channels, output_channels, kernel_size=kernel, stride=1, padding=padding, residual=True))
    
    
    return nn.Sequential(*layers)
  
def construct_decoder_layers(num_of_layers, input_channels, output_channels, first_layer_stride, add_spatial=False, kernel=3):
    layers = []
    padding = 1
    if kernel == 7:
      padding = 3
    # First layer
    layers.append(Conv2dTranspose(input_channels, output_channels, kernel_size=kernel, stride=first_layer_stride, padding=padding, output_padding=1))
    # Subsequent layers
    for _ in range(num_of_layers - 1):
        layers.append(Conv2d(output_channels, output_channels, kernel_size=kernel, stride=1, padding=padding, residual=True))
    
    if add_spatial:
      layers.append(SpatialAttention())
      
    return nn.Sequential(*layers)
  

      
class ResUNet384V4(nn.Module):
    def __init__(self):
        super(ResUNet384V4, self).__init__()
        
        self.betas = linear_schedule()
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.ellipse_params = {
            'center': (0.0, 0),  # (x,y)中心偏移(归一化坐标)
            'axes': (1, 0.75),      # (宽,高)比例
            'blur': 0.03             # 边缘模糊系数(相对于短边)
        }
        
        # --- Face Encoder ---
        self.face_encoder1_full = construct_encoder_layers(4, 12, 64, 1, kernel=3, add_spatial=True)
        self.face_pos_encoder1 = LearnablePositionalEncoding2D(d_model=64, max_h=384, max_w=384, dropout=0.1)
        self.fe_down1_full = construct_encoder_layers(4, 64, 64, 2, add_spatial=True)
        
        self.face_encoder2_full = construct_encoder_layers(4, 64, 128, 1, add_spatial=True)
        self.face_pos_encoder2 = LearnablePositionalEncoding2D(d_model=128, max_h=192, max_w=192, dropout=0.1) # After downsample
        self.fe_down2_full = construct_encoder_layers(4, 128, 128, 2, add_spatial=True)

        # Cross-Attention & Sparse Self-Attention for Feature Fusion
        # Dimensions for fed2 (face) and audio_emb
        # fed2: B, 128, H_face/4, W_face/4 (e.g., 384/4 = 96, so 96x96)
        # audio_emb: B, 256, 96, 96 (already adapted)
        
        # Cross-Attention: Face (Query) attends to Audio (Key/Value)
        # Query features: face (128 channels)
        # Key/Value features: audio (256 channels)
        # Output will have query_dim channels (128)
        self.face_audio_cross_attention = CrossAttentionBlock(
            query_dim=128, key_dim=256, value_dim=256, num_heads=8, dropout=0.1
        )
        
        # Adjust input channels for face_encoder3 if you concatenate the cross-attention output
        # If we add cross-attention output to face features:
        # face_encoder3 input remains 128 (from face features) + 256 (from audio) = 384
        # If we replace face features with attended features, then face_encoder3 input changes.
        # Let's concatenate for stronger fusion and adjust input channels.
        # Original: fed2_cat = torch.cat([fed2, audio_emb], dim=1) -> 128 + 256 = 384
        # Now: fed2_post_attn = attended_face_features (128)
        #      then cat [fed2_post_attn, audio_emb] -> 128 + 256 = 384. This works.

        # Sparse Self-Attention Block for combined features
        # After cross-attention and concatenation, the feature map might be quite rich.
        # Let's apply sparse self-attention after face_encoder3's output.
        # fed3 is (B, 256, H/8, W/8)
        
        # self.sparse_self_attention_block0 = SparseSelfAttentionBlock(
        #     in_channels=128, num_heads=8, window_size=(8, 8), dropout=0.1 # Example window size
        # )
        
        self.sparse_self_attention_block = SparseSelfAttentionBlock(
            in_channels=256, num_heads=8, window_size=(8, 8), dropout=0.1 # Example window size
        )
        
        self.sparse_self_attention_deface3_block = SparseSelfAttentionBlock(
            in_channels=320, num_heads=8, window_size=(8, 8), dropout=0.1 # Example window size
        )
        
        self.face_pos_encoder3 = LearnablePositionalEncoding2D(d_model=256, max_h=96, max_w=96, dropout=0.1) # After fe_down3 (384/8 = 48)

        self.face_encoder3 = construct_encoder_layers(3, 384, 256, 1, add_spatial=True) # Input channels still 384
        self.fe_down3 = construct_encoder_layers(3, 256, 256, 2, add_spatial=True)

        self.face_encoder4 = construct_encoder_layers(3, 256, 512, 1, add_spatial=True)
        self.face_pos_encoder4 = LearnablePositionalEncoding2D(d_model=512, max_h=48, max_w=48, dropout=0.1) # After fe_down4 (384/16 = 24)
        self.fe_down4 = construct_encoder_layers(3, 512, 512, 2)
        
        self.face_encoder5 = construct_encoder_layers(3, 512, 512, 1, add_spatial=True)
        self.face_pos_encoder5 = LearnablePositionalEncoding2D(d_model=512, max_h=24, max_w=24, dropout=0.1) # After fe_down5 (384/32 = 12)
        self.fe_down5 = construct_encoder_layers(3, 512, 512, 2)

        # --- Audio encoder ---
        self.audio_encoder1 = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
       
            Conv2d(32, 64, kernel_size=3, stride=(2, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=(2,1), padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            
            Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
        )
        self.audio_adapter1 = nn.AdaptiveAvgPool2d((96, 96)) # Target size for cross-attention
        self.audio_pos_encoder_ca = LearnablePositionalEncoding2D(d_model=256, max_h=96, max_w=96, dropout=0.1)


        self.bottlenet = construct_encoder_layers(3, 512, 512, 1, True)
        self.bottleneck_pos_encoder = LearnablePositionalEncoding2D(d_model=512, max_h=12, max_w=12, dropout=0.1) # 384/64 = 6
        
        # Decoders (channels adjusted for skip connections if needed)
        self.face_decoder5 = construct_decoder_layers(3, 512, 256, 2)
        self.fd_conv5 = construct_encoder_layers(3, 768, 256, 1) # 256 (deface5) + 512 (face5) = 768
        
        self.face_decoder4 = construct_decoder_layers(3, 256, 128, 2)
        self.fd_conv4 = construct_encoder_layers(3, 640, 320, 1) # 128 (deface4) + 512 (face4) = 640
        
        self.face_decoder3 = construct_decoder_layers(3, 320, 160, 2)
        self.fd_conv3 = construct_encoder_layers(3, 416, 256, 1) # 160 (deface3) + 256 (face3) = 416
        
        self.face_decoder2 = construct_decoder_layers(4, 256, 128, 2, add_spatial=True)
        self.fd_conv2 = construct_encoder_layers(4, 256, 128, 1, add_spatial=True) # 128 (deface2) + 128 (face2) = 256

        self.face_decoder1 = construct_decoder_layers(4, 128, 128, 2, add_spatial=True)
        self.face_decoder0 = construct_encoder_layers(4, 128, 128, 1, add_spatial=True) # This seems like an extra layer, no skip connection here
        
        self.fd_conv1 = construct_encoder_layers(4, 192, 64, 1) # 128 (deface1 from prev_decoder0) + 64 (face1) = 192

        self.output_block = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        
        #self.face_enhancer = FaceEnhencer()
        
    
    def generate_ellipse_mask(self, h, w, split_idx, device):
        """生成下半部分的椭圆遮罩"""
        y_bottom = torch.linspace(0, 1, h - split_idx, device=device) * 2 - 1
        x_coord = torch.linspace(-1, 1, w, device=device)
        y_grid, x_grid = torch.meshgrid(y_bottom, x_coord, indexing='ij')
        
        dx, dy = self.ellipse_params['center']
        a_ratio, b_ratio = self.ellipse_params['axes']
        
        adjusted_a = a_ratio * (w / h)  
        
        ellipse_mask = ((x_grid - dx)/adjusted_a)**2 + \
                      ((y_grid - dy)/b_ratio)**2 <= 1.0
        
        mask = ellipse_mask.float()
        if self.ellipse_params['blur'] > 0:
            kernel_size = int(min(h, w) * self.ellipse_params['blur']) | 1
            gaussian_blur_op = lambda x: GaussianBlur(kernel_size=(kernel_size, kernel_size), sigma=kernel_size/3)(x)
            mask = gaussian_blur_op(mask.unsqueeze(0).unsqueeze(0)).squeeze()
        
        return mask
      
    def diffuse(self, x, t, channels_to_mask=3):
        """
        带椭圆遮罩的扩散过程
        """
        b, c, h, w = x.shape
        split_idx = h // 2
        top_half = x[:, :, :split_idx, :]
        bottom_half = x[:, :, split_idx:, :]
        
        rgb_channels = bottom_half[:, :channels_to_mask, :, :]
        other_channels = bottom_half[:, channels_to_mask:, :, :]
        
        mask = self.generate_ellipse_mask(h, w, split_idx, x.device)
        mask = mask.unsqueeze(0).unsqueeze(0).repeat(b, channels_to_mask, 1, 1)
        
        sqrt_alpha_t = torch.sqrt(self.alphas_cumprod[t]).view(b, 1, 1, 1)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - self.alphas_cumprod[t]).view(b, 1, 1, 1)
        
        epsilon = torch.randn_like(rgb_channels)
        noisy_rgb = sqrt_alpha_t * rgb_channels + sqrt_one_minus_alpha_t * epsilon
        
        noisy_rgb = rgb_channels * (1 - mask) + noisy_rgb * mask
        
        noisy_bottom = torch.cat([noisy_rgb, other_channels], dim=1)
        result = torch.cat([top_half, noisy_bottom], dim=2)

        return result
      
    def sample_t(self, expanded_B, probabilities):
        probabilities = probabilities / probabilities.sum()
        t = torch.multinomial(probabilities, expanded_B, replacement=True)
        return t
                
    def forward(self, audio_sequences, face_sequences, step=None, train_face_enhancer=False):
        input_dim_size = len(face_sequences.size())
        B = audio_sequences.size(0)       
        
        if input_dim_size > 4:
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)

        expanded_B = face_sequences.size(0)
        t0 = self.sample_t(expanded_B, torch.tensor([1, 0, 0, 1], dtype=torch.float32)).to(face_sequences.device)
        self.alphas_cumprod = self.alphas_cumprod.to(face_sequences.device)
        face_sequences = self.diffuse(face_sequences.float(), t0, 3)
        
        # --- Audio encoding ---
        audio_emb = self.audio_encoder1(audio_sequences) # [B*T, 256, H_aud_enc, W_aud_enc]
        audio_emb = self.audio_adapter1(audio_emb)       # [B*T, 256, 96, 96]
        audio_emb = self.audio_pos_encoder_ca(audio_emb) # Positional encoding for audio features for cross-attention
        
        
        # --- Face encoding ---
        face1 = self.face_encoder1_full(face_sequences) # [B*T, 64, 384, 384]
        face1 = self.face_pos_encoder1(face1)           # Positional encoding
        fed1 = self.fe_down1_full(face1)                # [B*T, 64, 192, 192]
        
        
        face2 = self.face_encoder2_full(fed1)           # [B*T, 128, 192, 192]
        face2 = self.face_pos_encoder2(face2)           # Positional encoding
        fed2 = self.fe_down2_full(face2)                # [B*T, 128, 96, 96]

        
        # --- Cross-Attention ---
        # face (fed2) queries audio (audio_emb)
        # attended_face_features = self.face_audio_cross_attention(
        #     query_features=fed2, 
        #     key_value_features=audio_emb # audio_emb is already 96x96 spatial, same as fed2
        # )
        
        # Integrate attended features (e.g., add or concatenate)
        # Here, adding the attended features to the original face features
        #fed2_enhanced = fed2 + attended_face_features # Element-wise sum

        # Concatenate enhanced face features with original audio features
        # Assuming audio_emb is what was previously directly concatenated
        fed2_cat = torch.cat([fed2, audio_emb], dim=1) # [B*T, 128+256=384, 96, 96]

        face3 = self.face_encoder3(fed2_cat)            # [B*T, 256, 96, 96]
        face3 = self.face_pos_encoder3(face3)           # Positional encoding
        fed3 = self.fe_down3(face3)                     # [B*T, 256, 48, 48]

        # --- Sparse Self-Attention ---
        # Apply sparse self-attention on the combined features at this level
        fed3_attended = self.sparse_self_attention_block(fed3) # [B*T, 256, 48, 48]
        # You can add a residual connection here if sparse_self_attention_block doesn't handle it internally
        fed3 = fed3 + fed3_attended

        face4 = self.face_encoder4(fed3)       # [B*T, 512, 48, 48]
        face4 = self.face_pos_encoder4(face4)           # Positional encoding
        fed4 = self.fe_down4(face4)                     # [B*T, 512, 24, 24]

        face5 = self.face_encoder5(fed4)                # [B*T, 512, 24, 24]
        face5 = self.face_pos_encoder5(face5)           # Positional encoding
        fed5 = self.fe_down5(face5)                     # [B*T, 512, 12, 12]

        bottleneck_feat = self.bottlenet(fed5)          # [B*T, 512, 12, 12]
        bottleneck_feat = self.bottleneck_pos_encoder(bottleneck_feat) # Positional encoding

        # === Decoder ===
        deface5 = self.face_decoder5(bottleneck_feat)   # [B*T, 256, 24, 24]
        cat5 = torch.cat([deface5, face5], dim=1)       # [B*T, 256+512=768, 24, 24]
        cat5 = self.fd_conv5(cat5)                      # [B*T, 256, 24, 24]

        deface4 = self.face_decoder4(cat5)              # [B*T, 128, 48, 48]
        cat4 = torch.cat([deface4, face4], dim=1)       # [B*T, 128+512=640, 48, 48]
        cat4 = self.fd_conv4(cat4)                      # [B*T, 320, 48, 48]
        
        cat4_attended = self.sparse_self_attention_deface3_block(cat4)

        deface3 = self.face_decoder3(cat4 + cat4_attended)              # [B*T, 160, 96, 96]
        cat3 = torch.cat([deface3, face3], dim=1)       # [B*T, 160+256=416, 96, 96]
        cat3 = self.fd_conv3(cat3)                      # [B*T, 256, 96, 96]

        deface2 = self.face_decoder2(cat3)              # [B*T, 128, 192, 192]
        cat2 = torch.cat([deface2, face2], dim=1)       # [B*T, 128+128=256, 192, 192]
        cat2 = self.fd_conv2(cat2)                      # [B*T, 128, 192, 192]

        deface1 = self.face_decoder1(cat2)              # [B*T, 128, 384, 384]
        deface1 = self.face_decoder0(deface1)           # [B*T, 128, 384, 384] (no skip here, just further processing)
        cat1 = torch.cat([deface1, face1], dim=1)       # [B*T, 128+64=192, 384, 384]
        cat1 = self.fd_conv1(cat1)                      # [B*T, 64, 384, 384]

        x = self.output_block(cat1)                     # [B*T, 3, 384, 384]
        
        # if train_face_enhancer:
        #   ref_channels = face_sequences[:, 3:, :, :]
        #   concated = torch.cat([x, ref_channels], dim=1)
        #   outputs = self.face_enhancer(audio_emb, concated, fed2, fed3, deface2, deface1)
        
        if input_dim_size > 4:
            outputs = torch.split(x, B, dim=0)
            outputs = torch.stack(outputs, dim=2)
        else:
            outputs = x
            
            
        return outputs, None, None # Returning None for the projections as they are not used in the original forward.


class FaceEnhencer(nn.Module):
    def __init__(self):
        super(FaceEnhencer, self).__init__()
        
        
        # --- Face Encoder ---
        self.face_encoder1_full = construct_encoder_layers(3, 12, 32, 1, kernel=3)
        self.face_pos_encoder1 = LearnablePositionalEncoding2D(d_model=32, max_h=384, max_w=384, dropout=0.1)
        self.fe_down1_full = construct_encoder_layers(3, 32, 64, 2)
        
        self.face_encoder2_full = construct_encoder_layers(3, 64, 128, 1)
        self.face_pos_encoder2 = LearnablePositionalEncoding2D(d_model=128, max_h=192, max_w=192, dropout=0.1) # After downsample
        self.fe_down2_full = construct_encoder_layers(3, 128, 128, 2)

        self.face_audio_cross_attention = CrossAttentionBlock(
            query_dim=128, key_dim=256, value_dim=256, num_heads=8, dropout=0.1
        )
        
        self.sparse_self_attention_block = SparseSelfAttentionBlock(
            in_channels=256, num_heads=8, window_size=(8, 8), dropout=0.1 # Example window size
        )
        
        self.sparse_self_attention_deface3_block = SparseSelfAttentionBlock(
            in_channels=192, num_heads=8, window_size=(8, 8), dropout=0.1 # Example window size
        )
        
        self.face_pos_encoder3 = LearnablePositionalEncoding2D(d_model=256, max_h=96, max_w=96, dropout=0.1) # After fe_down3 (384/8 = 48)

        self.face_encoder3 = construct_encoder_layers(3, 384, 256, 1) # Input channels still 384
        self.fe_down3 = construct_encoder_layers(3, 256, 256, 2)

        self.face_encoder4 = construct_encoder_layers(3, 256, 256, 1)
        self.face_pos_encoder4 = LearnablePositionalEncoding2D(d_model=256, max_h=48, max_w=48, dropout=0.1) # After fe_down4 (384/16 = 24)
        self.fe_down4 = construct_encoder_layers(3, 256, 256, 2)
        
        self.face_encoder5 = construct_encoder_layers(3, 256, 256, 1)
        self.face_pos_encoder5 = LearnablePositionalEncoding2D(d_model=256, max_h=24, max_w=24, dropout=0.1) # After fe_down5 (384/32 = 12)
        self.fe_down5 = construct_encoder_layers(3, 256, 256, 2)


        self.bottlenet = construct_encoder_layers(3, 256, 256, 1, True)
        self.bottleneck_pos_encoder = LearnablePositionalEncoding2D(d_model=256, max_h=12, max_w=12, dropout=0.1) # 384/64 = 6
        
        # Decoders (channels adjusted for skip connections if needed)
        self.face_decoder5 = construct_decoder_layers(3, 256, 256, 2)
        self.fd_conv5 = construct_encoder_layers(3, 512, 256, 1) # 256 (deface5) + 512 (face5) = 768
        
        self.face_decoder4 = construct_decoder_layers(3, 256, 128, 2)
        self.fd_conv4 = construct_encoder_layers(3, 384, 192, 1) # 128 (deface4) + 512 (face4) = 640
        
        self.face_decoder3 = construct_decoder_layers(3, 192, 96, 2)
        self.fd_conv3 = construct_encoder_layers(3, 352, 256, 1) # 160 (deface3) + 256 (face3) = 416
        
        self.face_decoder2 = construct_decoder_layers(3, 256, 128, 2)
        self.fd_conv2 = construct_encoder_layers(3, 256, 128, 1) # 128 (deface2) + 128 (face2) = 256

        self.face_decoder1 = construct_decoder_layers(3, 128, 128, 2)
        self.face_decoder0 = construct_encoder_layers(3, 128, 128, 1) # This seems like an extra layer, no skip connection here
        
        self.fd_conv1 = construct_encoder_layers(3, 160, 64, 1) # 128 (deface1 from prev_decoder0) + 64 (face1) = 192

        self.output_block = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
    
    
                
    def forward(self, audio_sequences, face_sequences, prev_fed2, prev_fed3, prev_deface2, prev_deface1):
        input_dim_size = len(face_sequences.size())
        B = audio_sequences.size(0)       
        
        if input_dim_size > 4:
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)

        # --- Face encoding ---
        face1 = self.face_encoder1_full(face_sequences) # [B*T, 64, 384, 384]
        face1 = self.face_pos_encoder1(face1)           # Positional encoding
        fed1 = self.fe_down1_full(face1)                # [B*T, 64, 192, 192]
        
        
        face2 = self.face_encoder2_full(fed1)           # [B*T, 128, 192, 192]
        face2 = self.face_pos_encoder2(face2)           # Positional encoding
        fed2 = self.fe_down2_full(face2)                # [B*T, 128, 96, 96]
        fed2 = fed2 + prev_fed2

        
        fed2_cat = torch.cat([fed2, audio_sequences], dim=1) # [B*T, 128+256=384, 96, 96]

        face3 = self.face_encoder3(fed2_cat)            # [B*T, 256, 96, 96]
        face3 = self.face_pos_encoder3(face3)           # Positional encoding
        fed3 = self.fe_down3(face3)                     # [B*T, 256, 48, 48]
        fed3 = fed3 + prev_fed3

        # --- Sparse Self-Attention ---
        # Apply sparse self-attention on the combined features at this level
        fed3_attended = self.sparse_self_attention_block(fed3) # [B*T, 256, 48, 48]
        # You can add a residual connection here if sparse_self_attention_block doesn't handle it internally
        fed3 = fed3 + fed3_attended

        face4 = self.face_encoder4(fed3)       # [B*T, 512, 48, 48]
        face4 = self.face_pos_encoder4(face4)           # Positional encoding
        fed4 = self.fe_down4(face4)                     # [B*T, 512, 24, 24]

        face5 = self.face_encoder5(fed4)                # [B*T, 512, 24, 24]
        face5 = self.face_pos_encoder5(face5)           # Positional encoding
        fed5 = self.fe_down5(face5)                     # [B*T, 512, 12, 12]

        bottleneck_feat = self.bottlenet(fed5)          # [B*T, 512, 12, 12]
        bottleneck_feat = self.bottleneck_pos_encoder(bottleneck_feat) # Positional encoding

        # === Decoder ===
        deface5 = self.face_decoder5(bottleneck_feat)   # [B*T, 256, 24, 24]
        cat5 = torch.cat([deface5, face5], dim=1)       # [B*T, 256+512=768, 24, 24]
        cat5 = self.fd_conv5(cat5)                      # [B*T, 256, 24, 24]

        deface4 = self.face_decoder4(cat5)              # [B*T, 128, 48, 48]
        cat4 = torch.cat([deface4, face4], dim=1)       # [B*T, 128+512=640, 48, 48]
        cat4 = self.fd_conv4(cat4)                      # [B*T, 320, 48, 48]
        
        cat4_attended = self.sparse_self_attention_deface3_block(cat4)

        deface3 = self.face_decoder3(cat4 + cat4_attended)              # [B*T, 160, 96, 96]
        cat3 = torch.cat([deface3, face3], dim=1)       # [B*T, 160+256=416, 96, 96]
        cat3 = self.fd_conv3(cat3)                      # [B*T, 256, 96, 96]

        deface2 = self.face_decoder2(cat3)              # [B*T, 128, 192, 192]
        deface2 = deface2 + prev_deface2
        cat2 = torch.cat([deface2, face2], dim=1)       # [B*T, 128+128=256, 192, 192]
        cat2 = self.fd_conv2(cat2)                      # [B*T, 128, 192, 192]

        deface1 = self.face_decoder1(cat2)              # [B*T, 128, 384, 384]
        deface1 = deface1 + prev_deface1
        deface1 = self.face_decoder0(deface1)           # [B*T, 128, 384, 384] (no skip here, just further processing)
        cat1 = torch.cat([deface1, face1], dim=1)       # [B*T, 128+64=192, 384, 384]
        cat1 = self.fd_conv1(cat1)                      # [B*T, 64, 384, 384]

        x = self.output_block(cat1)                     # [B*T, 3, 384, 384]
        
            
        return x


