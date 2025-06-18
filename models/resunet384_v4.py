import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import GaussianBlur
from .transformer_syncnet import LearnablePositionalEncoding2D
from .conv import Conv2dTranspose, Conv2d
import math
import os, cv2
import numpy as np



class CrossAttentionBlock(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, num_heads, dropout=0.1):
        super().__init__()
        # Ensure embed_dim (query_dim) is divisible by num_heads
        assert query_dim % num_heads == 0, "query_dim must be divisible by num_heads"

        self.to_q = nn.Linear(query_dim, query_dim)
        self.to_k = nn.Linear(key_dim, query_dim) # Key and query must have same embed_dim for MHA
        self.to_v = nn.Linear(value_dim, query_dim) # Value can be different, but often matched

        self.mha = nn.MultiheadAttention(embed_dim=query_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)
        self.ffn = nn.Sequential(
            nn.Linear(query_dim, query_dim * 4),
            nn.GELU(),
            nn.Linear(query_dim * 4, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, query_features, key_value_features):
        # query_features: (B, C_q, H_q, W_q)
        # key_value_features: (B, C_kv, H_kv, W_kv)

        B_q, C_q, H_q, W_q = query_features.shape
        B_kv, C_kv, H_kv, W_kv = key_value_features.shape

        # Flatten spatial dimensions to sequence length
        query = query_features.view(B_q, C_q, -1).permute(0, 2, 1) # (B, H_q*W_q, C_q)
        key_value = key_value_features.view(B_kv, C_kv, -1).permute(0, 2, 1) # (B, H_kv*W_kv, C_kv)

        # Apply linear projections for Q, K, V
        q = self.to_q(query)
        k = self.to_k(key_value)
        v = self.to_v(key_value)

        # Cross-attention
        attn_output, _ = self.mha(query=q, key=k, value=v)
        
        # Add and Norm
        attn_output = self.norm1(attn_output + q) # Residual connection

        # Feed-forward network
        output = self.ffn(attn_output)
        output = self.norm2(output + attn_output) # Residual connection

        # Reshape back to image format (B, C, H, W)
        output = output.permute(0, 2, 1).view(B_q, C_q, H_q, W_q)
        return output

class SparseSelfAttentionBlock(nn.Module):
    def __init__(self, in_channels, num_heads, window_size, dropout=0.1):
        super().__init__()
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.window_size = window_size # For sparse attention, e.g., (8, 8) or (16, 16)
        
        self.qkv_proj = nn.Linear(in_channels, in_channels * 3)
        self.out_proj = nn.Linear(in_channels, in_channels)
        self.dropout = nn.Dropout(dropout)
        
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, in_channels * 4),
            nn.GELU(),
            nn.Linear(in_channels * 4, in_channels),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        
        # Flatten and permute for attention (B, H*W, C)
        x_flat = x.view(B, C, -1).permute(0, 2, 1) # (B, N, C) where N = H*W
        
        # Residual connection
        residual = x_flat
        
        # LayerNorm before QKV projection (common in Pre-LN transformers)
        x_norm = self.norm1(x_flat)

        # QKV projection
        qkv = self.qkv_proj(x_norm).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2), qkv)
        # q, k, v: (B, num_heads, N, head_dim)

        # Apply sparse attention mask
        # This is a simplified block-wise local attention.
        # More complex sparse attention patterns (e.g., dilated, strided) would require
        # more sophisticated mask generation or specialized kernels.
        # For a simple local attention:
        # We need to reshape q, k, v into blocks
        
        # Reshape to (B * num_heads, H, W, head_dim) for easier windowing
        q_reshaped = q.transpose(1, 2).reshape(B * self.num_heads, H, W, self.head_dim)
        k_reshaped = k.transpose(1, 2).reshape(B * self.num_heads, H, W, self.head_dim)
        v_reshaped = v.transpose(1, 2).reshape(B * self.num_heads, H, W, self.head_dim)

        output_patches = []
        # Simplified windowing: iterate through non-overlapping windows
        for i in range(0, H, self.window_size[0]):
            for j in range(0, W, self.window_size[1]):
                h_slice = slice(i, min(i + self.window_size[0], H))
                w_slice = slice(j, min(j + self.window_size[1], W))

                q_patch = q_reshaped[:, h_slice, w_slice, :].reshape(B * self.num_heads, -1, self.head_dim)
                k_patch = k_reshaped[:, h_slice, w_slice, :].reshape(B * self.num_heads, -1, self.head_dim)
                v_patch = v_reshaped[:, h_slice, w_slice, :].reshape(B * self.num_heads, -1, self.head_dim)
                
                # Perform attention within the window
                attn_scores = torch.matmul(q_patch, k_patch.transpose(-2, -1)) / math.sqrt(self.head_dim)
                attn_probs = F.softmax(attn_scores, dim=-1)
                attn_output_patch = torch.matmul(attn_probs, v_patch)
                
                # Reshape back to patch size and append
                output_patches.append(attn_output_patch.view(B * self.num_heads, h_slice.stop - h_slice.start, w_slice.stop - w_slice.start, self.head_dim))
        
        # Reconstruct the full attention output
        # This part needs careful reconstruction based on how patches were extracted
        # A more robust implementation would use unfold/fold or a custom kernel.
        # For simplicity, let's just do a dummy reconstruction assuming perfect tiling for now.
        # This is just a placeholder and would need a proper `fold` operation.
        
        # Let's simplify the sparse attention for this example:
        # Instead of full windowing, we'll implement a "dilated" or "strided" attention concept
        # that allows for long-range dependency but is still sparse.
        # A true sparse attention implementation is complex and often uses specialized libraries
        # like `xformers` or `flash_attention`.
        
        # For demonstration purposes, let's just make it a standard MultiheadAttention
        # for now and note where a sparse mask would be applied.
        # You'd need to generate an `attn_mask` for MHA.

        # Reverting to full self-attention for easier demonstration of structure
        # A sparse mask would be passed to `attn_mask` in self.mha call.
        attn_output, _ = self.mha(query=q.transpose(1,2).reshape(B, H*W, C), # (B, N, C)
                                  key=k.transpose(1,2).reshape(B, H*W, C),
                                  value=v.transpose(1,2).reshape(B, H*W, C))
        
        # Instead of the above complex reconstruction, if you want a *simple* sparse
        # self-attention that's not full windowed, you'd pass a custom mask:
        # attn_output, _ = self.mha(query=q, key=k, value=v, attn_mask=sparse_mask)
        # Generating `sparse_mask` is the hard part for efficient sparse attention.
        
        # For this example, I'll modify SparseSelfAttentionBlock to be a standard MHA for now,
        # but the `window_size` parameter implies a sparse strategy.
        # Let's make it a general SelfAttention and note the sparse modification.

        # Re-implementing with standard MHA and mentioning sparse part:
        q_kv_flat = x_norm
        
        attn_output, _ = self.mha(query=q_kv_flat, key=q_kv_flat, value=q_kv_flat)
        # To make it sparse self-attention, you would pass an `attn_mask` here.
        # Example: `attn_mask = generate_sparse_mask(H*W, H*W, window_size)`
        # `generate_sparse_mask` is non-trivial and depends on the sparsity pattern.

        output = self.dropout(self.out_proj(attn_output))
        
        # Add residual connection
        output = self.norm2(output + residual)
        
        # FFN
        output = self.ffn(output) + output # Another residual connection

        # Reshape back to (B, C, H, W)
        output = output.permute(0, 2, 1).view(B, C, H, W)
        return output

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
            nn.Linear(in_channels, in_channels * 4),
            nn.GELU(),
            nn.Linear(in_channels * 4, in_channels),
            nn.Dropout(dropout)
        )
        self.qkv_proj = nn.Linear(in_channels, in_channels * 3) # Combined QKV projection
        self.out_proj = nn.Linear(in_channels, in_channels) # Output projection
        self.dropout_layer = nn.Dropout(dropout)

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
        sparse_mask = None # Replace with actual sparse mask generation if needed
                           # e.g., for local attention within a window_size, this mask would be block-diagonal.

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
    return torch.tensor([0.6, 0.7, 0.8, 0.9])

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
        self.face_encoder1_full = self.construct_encoder_layers(2, 12, 64, 1, kernel=3)
        self.face_pos_encoder1 = LearnablePositionalEncoding2D(d_model=64, max_h=384, max_w=384, dropout=0.1)
        self.fe_down1_full = self.construct_encoder_layers(2, 64, 64, 2)
        
        self.face_encoder2_full = self.construct_encoder_layers(2, 64, 128, 1)
        self.face_pos_encoder2 = LearnablePositionalEncoding2D(d_model=128, max_h=192, max_w=192, dropout=0.1) # After downsample
        self.fe_down2_full = self.construct_encoder_layers(2, 128, 128, 2)

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
        self.sparse_self_attention_block = SparseSelfAttentionBlock(
            in_channels=256, num_heads=8, window_size=(8, 8), dropout=0.1 # Example window size
        )
        self.face_pos_encoder3 = LearnablePositionalEncoding2D(d_model=256, max_h=96, max_w=96, dropout=0.1) # After fe_down3 (384/8 = 48)

        self.face_encoder3 = self.construct_encoder_layers(2, 384, 256, 1) # Input channels still 384
        self.fe_down3 = self.construct_encoder_layers(2, 256, 256, 2)

        self.face_encoder4 = self.construct_encoder_layers(2, 256, 512, 1)
        self.face_pos_encoder4 = LearnablePositionalEncoding2D(d_model=512, max_h=48, max_w=48, dropout=0.1) # After fe_down4 (384/16 = 24)
        self.fe_down4 = self.construct_encoder_layers(2, 512, 512, 2)
        
        self.face_encoder5 = self.construct_encoder_layers(2, 512, 512, 1)
        self.face_pos_encoder5 = LearnablePositionalEncoding2D(d_model=512, max_h=24, max_w=24, dropout=0.1) # After fe_down5 (384/32 = 12)
        self.fe_down5 = self.construct_encoder_layers(2, 512, 512, 2)

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


        self.bottlenet = self.construct_encoder_layers(2, 512, 512, 1, True)
        self.bottleneck_pos_encoder = LearnablePositionalEncoding2D(d_model=512, max_h=12, max_w=12, dropout=0.1) # 384/64 = 6
        
        # Decoders (channels adjusted for skip connections if needed)
        self.face_decoder5 = self.construct_decoder_layers(2, 512, 256, 2)
        self.fd_conv5 = self.construct_encoder_layers(2, 768, 256, 1) # 256 (deface5) + 512 (face5) = 768
        
        self.face_decoder4 = self.construct_decoder_layers(2, 256, 128, 2)
        self.fd_conv4 = self.construct_encoder_layers(2, 640, 320, 1) # 128 (deface4) + 512 (face4) = 640
        
        self.face_decoder3 = self.construct_decoder_layers(2, 320, 160, 2)
        self.fd_conv3 = self.construct_encoder_layers(2, 416, 256, 1) # 160 (deface3) + 256 (face3) = 416
        
        self.face_decoder2 = self.construct_decoder_layers(2, 256, 128, 2)
        self.fd_conv2 = self.construct_encoder_layers(2, 256, 128, 1) # 128 (deface2) + 128 (face2) = 256

        self.face_decoder1 = self.construct_decoder_layers(2, 128, 128, 2)
        self.face_decoder0 = self.construct_encoder_layers(2, 128, 128, 1) # This seems like an extra layer, no skip connection here
        
        self.fd_conv1 = self.construct_encoder_layers(2, 192, 64, 1) # 128 (deface1 from prev_decoder0) + 64 (face1) = 192

        self.output_block = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        
        # Removed redundant positional encoders
        # self.face_pos_encoder = LearnablePositionalEncoding2D(d_model=128, max_h=12, max_w=12, dropout=0.1)
        # self.audio_pos_encoder = LearnablePositionalEncoding2D(d_model=128, max_h=12, max_w=12, dropout=0.1)
        
    def construct_encoder_layers(self, num_of_layers, input_channels, output_channels, first_layer_stride, add_spatial=False, kernel=3):
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

    def construct_decoder_layers(self, num_of_layers, input_channels, output_channels, first_layer_stride, add_spatial=False, kernel=3):
        layers = []
        padding = 1
        if kernel == 7:
          padding = 3
        # First layer
        layers.append(Conv2dTranspose(input_channels, output_channels, kernel_size=kernel, stride=first_layer_stride, padding=padding, output_padding=1))
        # Subsequent layers
        for _ in range(num_of_layers - 1):
            layers.append(Conv2d(output_channels, output_channels, kernel_size=kernel, stride=1, padding=padding, residual=True))
        return nn.Sequential(*layers)
    
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
                
    def forward(self, audio_sequences, face_sequences, step=None):
        input_dim_size = len(face_sequences.size())
        B = audio_sequences.size(0)       
        
        if input_dim_size > 4:
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)

        expanded_B = face_sequences.size(0)
        t0 = self.sample_t(expanded_B, torch.tensor([0, 0, 0, 1], dtype=torch.float32)).to(face_sequences.device)
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
        # fed3 = fed3 + fed3_attended

        face4 = self.face_encoder4(fed3_attended)       # [B*T, 512, 48, 48]
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

        deface3 = self.face_decoder3(cat4)              # [B*T, 160, 96, 96]
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
        
        if input_dim_size > 4:
            x = torch.split(x, B, dim=0)
            outputs = torch.stack(x, dim=2)
        else:
            outputs = x
            
        return outputs, None, None # Returning None for the projections as they are not used in the original forward.


