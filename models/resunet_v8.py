import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.transforms import GaussianBlur
from .transformer_syncnet import LearnablePositionalEncoding2D
from .spatial_attention import SpatialAttention
from .conv import Conv2dTranspose, Conv2d
import math
import os, cv2
import numpy as np
from torchvision.transforms import GaussianBlur


def construct_encoder_layers(num_of_layers, input_channels, output_channels, first_layer_stride, kernel=3):
    layers = []
    padding = 1
    if kernel == 7:
      padding = 3
    elif kernel == 5:
      padding = 2
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


class WindowCrossAttention(nn.Module):
    """Improved window-based cross attention with stable fusion to avoid gradient vanishing.

    - Normalizes Q/K/V per-window before attention.
    - Uses separate q/k/v projections.
    - Pads spatial dims so H,W don't need to be divisible by window_size.
    - After attention, uses a small conv-MLP and residual add to visual features.
    """
    def __init__(self, visual_dim, audio_dim, window_size=8, num_heads=4, attn_dropout=0.0, proj_dropout=0.0):
        super().__init__()
        self.visual_dim = visual_dim
        self.window_size = window_size
        self.num_heads = num_heads

        # Separate projections
        self.q_proj = nn.Conv2d(visual_dim, visual_dim, kernel_size=1, bias=True)
        self.k_proj = nn.Conv2d(audio_dim, visual_dim, kernel_size=1, bias=True)
        self.v_proj = nn.Conv2d(audio_dim, visual_dim, kernel_size=1, bias=True)

        # PyTorch multihead expects [B, L, E] when batch_first=True
        self.attention = nn.MultiheadAttention(embed_dim=visual_dim, num_heads=num_heads, dropout=attn_dropout, batch_first=True)

        # Small conv-MLP applied to attention output (channel-preserving)
        self.post_conv = nn.Sequential(
            nn.Conv2d(visual_dim, visual_dim, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(visual_dim, visual_dim, kernel_size=1, bias=True),
        )

        # Normalization: we apply LayerNorm over channel dim after reshaping windows to [..., C]
        # We'll create a small helper LayerNorm; will be applied per-window sequence tokens (last dim = C)
        self.window_ln_q = nn.LayerNorm(visual_dim)
        self.window_ln_k = nn.LayerNorm(visual_dim)
        self.window_ln_v = nn.LayerNorm(visual_dim)

        # Final LN over channels (applied on [B, C, H, W] by permuting)
        self.final_ln = nn.GroupNorm(1, visual_dim)  # GroupNorm(1, C) ~ InstanceNorm across spatial but stable

        # Optional projection dropout after conv-mlp
        self.proj_dropout = nn.Dropout(proj_dropout) if proj_dropout > 0 else nn.Identity()

        # Learnable scaling factor for residual connection
        self.residual_scale = nn.Parameter(torch.ones(1))

    def pad_if_needed(self, x, window_size):
        # pad on H and W if not divisible by window_size
        B, C, H, W = x.shape
        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        if pad_h == 0 and pad_w == 0:
            return x, 0, 0
        x = F.pad(x, (0, pad_w, 0, pad_h))  # pad (left,right, top,bottom)
        return x, pad_h, pad_w

    def window_partition(self, x, window_size):
        """Partition into non-overlapping windows. Returns (windows, H_pad, W_pad, Hp, Wp)"""
        B, C, H, W = x.shape
        x, pad_h, pad_w = self.pad_if_needed(x, window_size)
        _, _, Hp, Wp = x.shape
        # reshape: B, C, H//ws, ws, W//ws, ws
        x = x.view(B, C, Hp // window_size, window_size, Wp // window_size, window_size)
        # permute to (B, H//ws, W//ws, ws, ws, C)
        windows = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        # collapse to (num_windows_total*B, ws*ws, C)
        windows = windows.view(-1, window_size * window_size, C)
        return windows, Hp, Wp, pad_h, pad_w

    def window_reverse(self, windows, window_size, Hp, Wp, pad_h, pad_w):
        """Reverse windows into padded spatial layout, then unpad to original H,W."""
        # windows: (num_windows_total*B, ws*ws, C)
        B = int(windows.shape[0] // ((Hp // window_size) * (Wp // window_size)))
        x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, Hp, Wp)
        if pad_h != 0 or pad_w != 0:
            x = x[:, :, : Hp - pad_h, : Wp - pad_w]
        return x

    def forward(self, visual_feat, audio_feat):
        """
        visual_feat: [B, C_v, H, W]
        audio_feat:  [B, C_a, H_a, W_a] (will be resized if necessary)
        returns: fused [B, C_v, H, W]
        """
        B, C_v, H, W = visual_feat.shape

        # Store original visual features for residual connection
        visual_identity = visual_feat.clone()

        # Resize audio to visual spatial dims (bilinear)
        if audio_feat.shape[2:] != (H, W):
            audio_feat = F.interpolate(audio_feat, size=(H, W), mode='bilinear', align_corners=False)

        # Project to q/k/v (conv 1x1)
        Q = self.q_proj(visual_feat)      # [B, C_v, H, W]
        K = self.k_proj(audio_feat)       # [B, C_v, H, W]
        V = self.v_proj(audio_feat)       # [B, C_v, H, W]

        # Partition into windows (with padding if needed)
        window_size = self.window_size
        Qw, Hp, Wp, pad_h, pad_w = self.window_partition(Q, window_size)
        Kw, _, _, _, _ = self.window_partition(K, window_size)
        Vw, _, _, _, _ = self.window_partition(V, window_size)
        # Qw/Kw/Vw: [num_windows*B, ws*ws, C_v]

        # Normalise per-window (LayerNorm expects [..., C])
        Qw = self.window_ln_q(Qw)
        Kw = self.window_ln_k(Kw)
        Vw = self.window_ln_v(Vw)

        # MultiheadAttention expects (B_batch, L_q, E), we're already in that shape
        # Use attn with Qw as queries, Kw as keys, Vw as values
        attn_out, _ = self.attention(Qw, Kw, Vw)  # -> [num_windows*B, ws*ws, C_v]

        # Convert attn_out back to spatial windows
        attn_spatial = self.window_reverse(attn_out, window_size, Hp, Wp, pad_h, pad_w)  # [B, C_v, H, W] possibly padded

        # Post processing conv-MLP, dropout and residual add to attention output
        processed_audio = attn_spatial + self.post_conv(attn_spatial)
        processed_audio = self.proj_dropout(processed_audio)

        # Residual fusion: visual + processed_audio with learnable scaling
        # This ensures gradient can flow through both paths
        fused = visual_identity + self.residual_scale * processed_audio

        # Final normalization for stability - apply to the fused output
        #fused = self.final_ln(fused)

        return fused



class ResUNet384V8(nn.Module):
    def __init__(self, print_gradients=False):
        super(ResUNet384V8, self).__init__()
        
        self.print_gradients = print_gradients
        self.gradient_hooks = []
        
        self.ellipse_params = {
            'center': (0.0, 0),  # (x,y)中心偏移(归一化坐标)
            'axes': (1, 0.75),      # (宽,高)比例
            'blur': 0.0             # 边缘模糊系数(相对于短边)
        }
        
        # --- First UNet (processes bottom half) ---
        self.bottom_unet_encoder1 = construct_encoder_layers(3, 6, 24, 1)
        self.bottom_unet_down1 = construct_encoder_layers(3, 24, 48, 2)
                
        self.bottom_unet_encoder2 = construct_encoder_layers(3, 48, 96, 1)
        self.bottom_unet_down2 = construct_encoder_layers(3, 96, 96, 2)
                       
        self.bottom_unet_encoder3 = construct_encoder_layers(3, 96, 192, 1)
        self.bottom_unet_down3 = construct_encoder_layers(3, 192, 192, 2)

        self.bottom_unet_encoder4 = construct_encoder_layers(2, 192, 192, 1)
        self.bottom_unet_pos_encoder4 = LearnablePositionalEncoding2D(d_model=192, max_h=24, max_w=48, dropout=0.1)
        self.bottom_unet_down4 = construct_encoder_layers(2, 192, 192, 2)
        
        self.bottom_unet_encoder5 = construct_encoder_layers(2, 192, 192, 1)
        self.bottom_unet_pos_encoder5 = LearnablePositionalEncoding2D(d_model=192, max_h=12, max_w=24, dropout=0.1)
        self.bottom_unet_down5 = construct_encoder_layers(2, 192, 192, 2)

        # --- MaxPooling for residual connection from encoder1 to encoder3 ---
        self.encoder1_to_encoder3_pool = nn.MaxPool2d(kernel_size=4, stride=4)  # 4x downsampling
        self.encoder1_to_encoder3_conv = nn.Conv2d(24, 192, kernel_size=1)  # Channel adjustment
        

        # --- MaxPooling for residual connection from encoder1 to encoder4 ---
        self.encoder1_to_encoder4_pool = nn.MaxPool2d(kernel_size=8, stride=8)  # 8x downsampling
        self.encoder1_to_encoder4_conv = nn.Conv2d(24, 192, kernel_size=1)  # Channel adjustment
        
        self.encoder2_to_encoder4_pool = nn.MaxPool2d(kernel_size=4, stride=4)  # 8x downsampling
        self.encoder2_to_encoder4_conv = nn.Conv2d(96, 192, kernel_size=1)  # Channel adjustment
        
        
        self.encoder3_to_encoder5_pool = nn.MaxPool2d(kernel_size=4, stride=4)  # 4x downsampling
        self.encoder3_to_encoder5_conv = nn.Conv2d(192, 192, kernel_size=1)  # Channel adjustment

        # --- Audio encoder ---
        self.audio_encoder1 = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
       
            Conv2d(32, 64, kernel_size=3, stride=(2, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=(2,1), padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            
            Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
        )
                

        self.bottom_unet_bottleneck = construct_encoder_layers(3, 192, 256, 1)
        self.bottom_unet_bottleneck_pos_encoder = LearnablePositionalEncoding2D(d_model=256, max_h=6, max_w=12, dropout=0.1)
        
        # Audio adapters for fusing with bottom decoders at multiple levels
        self.audio_adapter_bottleneck = nn.AdaptiveAvgPool2d((6, 12))  # Match bottleneck spatial dims
        self.audio_adapter_5 = nn.AdaptiveAvgPool2d((12, 24))   # Match decoder5 spatial dims
        self.audio_adapter_5_conv = nn.Conv2d(256, 256, kernel_size=1)  # Channel adjustment
        
        self.audio_adapter_4 = nn.AdaptiveAvgPool2d((24, 48))   # Match decoder4 spatial dims
        self.audio_adapter_4_conv = nn.Conv2d(256, 256, kernel_size=1)  # Channel adjustment
        
        self.audio_adapter_3 = nn.AdaptiveAvgPool2d((48, 96))   # Match decoder3 spatial dims  
        self.audio_adapter_3_conv = nn.Conv2d(256, 256, kernel_size=1)  # Channel adjustment
        
        self.audio_adapter_2 = nn.AdaptiveAvgPool2d((96, 192))   # Match decoder2 spatial dims
        self.audio_adapter_2_conv = nn.Conv2d(256, 256, kernel_size=1)  # Channel adjustment
        
        # Positional encoders for audio at different scales
        self.audio_pos_encoder_bottleneck = LearnablePositionalEncoding2D(d_model=256, max_h=6, max_w=12, dropout=0.1)
        self.audio_pos_encoder_5 = LearnablePositionalEncoding2D(d_model=256, max_h=12, max_w=24, dropout=0.1)
        self.audio_pos_encoder_4 = LearnablePositionalEncoding2D(d_model=256, max_h=24, max_w=48, dropout=0.1)
        self.audio_pos_encoder_3 = LearnablePositionalEncoding2D(d_model=256, max_h=48, max_w=96, dropout=0.1)
        self.audio_pos_encoder_2 = LearnablePositionalEncoding2D(d_model=256, max_h=96, max_w=192, dropout=0.1)
        
        # WindowCrossAttention fusion modules for different levels
        # Using different window sizes based on feature map sizes for optimal memory usage
        self.av_fusion_2 = WindowCrossAttention(96, 256, window_size=8, num_heads=4)    # Larger window for smaller features
        self.av_fusion_3 = WindowCrossAttention(192, 256, window_size=8, num_heads=4)   # Medium window
        self.av_fusion_4 = WindowCrossAttention(192, 256, window_size=4, num_heads=4)   # Smaller window for larger features
        self.av_fusion_5 = WindowCrossAttention(192, 256, window_size=4, num_heads=4)   # Smaller window
        self.av_fusion_bottleneck = WindowCrossAttention(256, 256, window_size=2, num_heads=4)  # Smallest window for bottleneck
       
                
        # Bottom UNet Decoders
        self.bottom_unet_decoder5 = construct_decoder_layers(3, 256, 256, 2)  # Only visual features
        self.bottom_unet_conv5 = construct_encoder_layers(3, 448, 320, 1) # 256 (debottom5) + 192 (bottom5) = 448
        
        self.bottom_unet_decoder4 = construct_decoder_layers(3, 320, 256, 2)
        self.bottom_unet_conv4 = construct_encoder_layers(3, 448, 256, 1) # 256 (debottom4) + 192 (bottom4) = 448
        
        self.bottom_unet_decoder3 = construct_decoder_layers(3, 256, 256, 2)
        self.bottom_unet_conv3 = construct_encoder_layers(3, 448, 256, 1) # 256 (debottom3) + 192 (bottom3) = 448
        
        self.bottom_unet_decoder2 = construct_decoder_layers(3, 256, 128, 2, add_spatial=True)
        self.bottom_unet_conv2 = construct_encoder_layers(3, 224, 128, 1) # 128 (debottom2) + 96 (bottom2) = 224

        self.bottom_unet_decoder1 = construct_decoder_layers(3, 128, 64, 2, add_spatial=True)
        self.bottom_unet_decoder0 = construct_encoder_layers(3, 64, 32, 1)
                
        self.bottom_unet_conv1 = construct_encoder_layers(3, 56, 32, 1) # 32 (debottom1 from prev_decoder0) + 24 (bottom1) = 56

        self.bottom_unet_output_block = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        
                
    def register_gradient_hook(self, tensor, name):
        """Register a hook to print gradients for a tensor."""
        def hook_fn(grad):
            if self.print_gradients:
                grad_norm = grad.norm().item()
                print(f"Gradient norm for {name}: {grad_norm}")
                # Also print some statistics about the gradient
                print(f"  Mean: {grad.mean().item():.6f}, Std: {grad.std().item():.6f}")
                print(f"  Min: {grad.min().item():.6f}, Max: {grad.max().item():.6f}")
        handle = tensor.register_hook(hook_fn)
        self.gradient_hooks.append(handle)
        return tensor
        
    def clear_gradient_hooks(self):
        """Remove all registered gradient hooks."""
        for handle in self.gradient_hooks:
            handle.remove()
        self.gradient_hooks.clear()  
    
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
      
    def diffuse(self, x, channels_to_mask=3):
        """
        带椭圆遮罩的扩散过程 —— 使用黑色（0）填充遮罩区域，而非噪声
        """
        b, c, h, w = x.shape
        split_idx = h // 2
        top_half = x[:, :, :split_idx, :]
        bottom_half = x[:, :, split_idx:, :]
        
        rgb_channels = bottom_half[:, :channels_to_mask, :, :]
        other_channels = bottom_half[:, channels_to_mask:, :, :]
        
        # 生成椭圆遮罩
        mask = self.generate_ellipse_mask(h, w, split_idx, x.device)
        mask = mask.unsqueeze(0).unsqueeze(0).repeat(b, channels_to_mask, 1, 1)
        
        # 将遮罩区域设为黑色（0），其余部分保持原样
        masked_rgb = rgb_channels * (1 - mask)  # 非遮罩区域保留原值
        # 遮罩区域直接设为 0（黑色）

        noisy_bottom = torch.cat([masked_rgb, other_channels], dim=1)
        result = torch.cat([top_half, noisy_bottom], dim=2)

        return result

                
    def apply_global_gaussian_blur(self, image, kernel_size=15, sigma=5.0):
        """
        Apply Gaussian blur to the entire input image
        kernel_size: should be odd integer, larger = more blur
        sigma: standard deviation for Gaussian kernel
        """
        # Ensure kernel_size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Create Gaussian blur transform
        blur_transform = GaussianBlur(kernel_size=(kernel_size, kernel_size), 
                                    sigma=(sigma, sigma))
        
        return blur_transform(image)
      
      
    def forward(self, audio_sequences, face_sequences, step=None, training=True):
        
        input_dim_size = len(face_sequences.size())
        B = audio_sequences.size(0)       
        
        if input_dim_size > 4:
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)
            
        original_top_half = face_sequences[:, :3, :192, :]
        original_bottom_half = face_sequences[:, :3, 192:, :]

        face_sequences = F.normalize(face_sequences, p=2, dim=1)
        audio_sequences = F.normalize(audio_sequences, p=2, dim=1)
    
        # Process audio sequences
        audio_embedding = self.audio_encoder1(audio_sequences)
                                
        # Extract first 3 channels for processing
        face_sequences_3ch = face_sequences[:, :3, :, :]
        
        # Apply diffusion to the face sequences
        face_sequences_3ch = self.diffuse(face_sequences_3ch, channels_to_mask=3)
        
        # Extract remaining channels as reference
        face_sequences_ref = face_sequences[:, 3:, :, :]
        
        # Prepare audio features for multiple fusion points
        audio_bottleneck = self.audio_adapter_bottleneck(audio_embedding)
        audio_bottleneck = self.audio_pos_encoder_bottleneck(audio_bottleneck)
        
        audio_5 = self.audio_adapter_5(audio_embedding)
        audio_5 = self.audio_adapter_5_conv(audio_5)
        audio_5 = self.audio_pos_encoder_5(audio_5)
        
        audio_4 = self.audio_adapter_4(audio_embedding)
        audio_4 = self.audio_adapter_4_conv(audio_4)
        audio_4 = self.audio_pos_encoder_4(audio_4)
        
        audio_3 = self.audio_adapter_3(audio_embedding)
        audio_3 = self.audio_adapter_3_conv(audio_3)
        audio_3 = self.audio_pos_encoder_3(audio_3)
        
        audio_2 = self.audio_adapter_2(audio_embedding)
        audio_2 = self.audio_adapter_2_conv(audio_2)
        audio_2 = self.audio_pos_encoder_2(audio_2)
        
        
        # Get image dimensions
        _, _, h, w = face_sequences_3ch.shape
        split_idx = h // 2
        
        # Extract 9 reference channels for bottom encoder
        bottom_ref_channels = face_sequences_ref[:, :, split_idx:, :]
        
        # Split into top and bottom halves
        top_half = face_sequences_3ch[:, :, :split_idx, :]
        bottom_half = face_sequences_3ch[:, :, split_idx:, :]
        ref_bottom_half = face_sequences_ref[:, :, split_idx:, :]
        
        ref_residual = torch.cat([top_half, ref_bottom_half], dim=2)
        
        # Combine bottom half with reference channels for bottom encoder
        bottom_half_with_ref = torch.cat([bottom_half, bottom_ref_channels], dim=1)
        
        # First UNet: Process bottom half to generate bottom half output
        # Encode bottom half through bottom UNet
        bottom_enc1 = self.bottom_unet_encoder1(bottom_half_with_ref)
        # Create residual connection from encoder1 to encoder3
        encoder1_to_encoder3_residual = self.encoder1_to_encoder3_pool(bottom_enc1)  # Downsample spatially
        encoder1_to_encoder3_residual = self.encoder1_to_encoder3_conv(encoder1_to_encoder3_residual)  # Adjust channels
        
        # Create residual connection from encoder1 to encoder4
        encoder1_to_encoder4_residual = self.encoder1_to_encoder4_pool(bottom_enc1)  # Downsample spatially
        encoder1_to_encoder4_residual = self.encoder1_to_encoder4_conv(encoder1_to_encoder4_residual)  # Adjust channels
        
        bottom_down1 = self.bottom_unet_down1(bottom_enc1)
        
        bottom_enc2 = self.bottom_unet_encoder2(bottom_down1)
        encoder2_to_encoder4_residual = self.encoder2_to_encoder4_pool(bottom_enc2)  # Downsample spatially
        encoder2_to_encoder4_residual = self.encoder2_to_encoder4_conv(encoder2_to_encoder4_residual)  # Adjust channels
        bottom_down2 = self.bottom_unet_down2(bottom_enc2)
                
        bottom_enc3 = self.bottom_unet_encoder3(bottom_down2)      
        # Fuse audio with visual features at encoder3 level using WindowCrossAttention
        bottom_enc3_fused = self.av_fusion_3(bottom_enc3, audio_3)
        encoder3_to_encoder5_residual = self.encoder3_to_encoder5_pool(bottom_enc3_fused)
        encoder3_to_encoder5_residual = self.encoder3_to_encoder5_conv(encoder3_to_encoder5_residual)  # Adjust channels
        
        # Register gradient hook for bottom_enc3 if print_gradients is enabled
        if self.print_gradients:
            bottom_enc3_fused = self.register_gradient_hook(bottom_enc3_fused, "bottom_enc3_after_av_fusion_3")
        
        # Add the residual connection from encoder1
        bottom_enc3_fused = bottom_enc3_fused + encoder1_to_encoder3_residual
        bottom_enc3_fused = self.bottom_unet_down3(bottom_enc3_fused)
        # Register gradient hook for bottom_down3 if print_gradients is enabled
        if self.print_gradients:
            bottom_enc3_fused = self.register_gradient_hook(bottom_enc3_fused, "bottom_down3_after_bottom_unet_down3")
        
        
        bottom_enc4 = self.bottom_unet_encoder4(bottom_enc3_fused + encoder2_to_encoder4_residual)
        # Add the residual connection from encoder1
        bottom_enc4 = bottom_enc4 + encoder1_to_encoder4_residual
        # Fuse audio with visual features at encoder4 level using WindowCrossAttention
        bottom_enc4 = self.av_fusion_4(bottom_enc4, audio_4)
        bottom_down4 = self.bottom_unet_down4(bottom_enc4 + encoder1_to_encoder4_residual)
        
        bottom_enc5 = self.bottom_unet_encoder5(bottom_down4)
        
        bottom_enc5 += encoder3_to_encoder5_residual
        
        # Fuse audio with visual features at encoder5 level using WindowCrossAttention
        bottom_enc5 = self.av_fusion_5(bottom_enc5, audio_5)
        bottom_down5 = self.bottom_unet_down5(bottom_enc5 + encoder3_to_encoder5_residual)
        
        bottom_bottleneck = self.bottom_unet_bottleneck(bottom_down5)
        bottom_bottleneck = self.bottom_unet_bottleneck_pos_encoder(bottom_bottleneck)
        # Fuse audio with visual features at bottleneck level using WindowCrossAttention
        bottom_bottleneck = self.av_fusion_bottleneck(bottom_bottleneck, audio_bottleneck)
        
        # Decode to generate bottom half output with multi-level audio fusion
        bottom_de5 = self.bottom_unet_decoder5(bottom_bottleneck)
        
        bottom_cat5 = torch.cat([bottom_de5, bottom_enc5], dim=1)
        bottom_cat5 = self.bottom_unet_conv5(bottom_cat5)
        
        # Fuse audio at decoder4 level
        bottom_de4 = self.bottom_unet_decoder4(bottom_cat5)
        
        bottom_cat4 = torch.cat([bottom_de4, bottom_enc4], dim=1)
        bottom_cat4 = self.bottom_unet_conv4(bottom_cat4)
        
        # Fuse audio at decoder3 level
        bottom_de3 = self.bottom_unet_decoder3(bottom_cat4)
        bottom_cat3 = torch.cat([bottom_de3, bottom_enc3], dim=1)
        bottom_cat3 = self.bottom_unet_conv3(bottom_cat3)
        
        # Fuse audio at decoder2 level
        bottom_de2 = self.bottom_unet_decoder2(bottom_cat3)
        bottom_cat2 = torch.cat([bottom_de2, bottom_enc2], dim=1)
        bottom_cat2 = self.bottom_unet_conv2(bottom_cat2)
        
        bottom_de1 = self.bottom_unet_decoder1(bottom_cat2)
        bottom_de0 = self.bottom_unet_decoder0(bottom_de1)
                
        bottom_cat1 = torch.cat([bottom_de0, bottom_enc1], dim=1)
        bottom_cat1 = self.bottom_unet_conv1(bottom_cat1)
        
        generated_bottom_half = self.bottom_unet_output_block(bottom_cat1)
        
        # Use detached version for enhancement
        combined_full_image = torch.cat([original_top_half, generated_bottom_half], dim=2)
        
        if input_dim_size > 4:            
            bottom_outputs = torch.split(generated_bottom_half, B, dim=0)
            bottom_outputs = torch.stack(bottom_outputs, dim=2)
            original_top_half = torch.split(original_top_half, B, dim=0)
            original_top_half = torch.stack(original_top_half, dim=2)
        else:
            bottom_outputs = generated_bottom_half
                        
        
        if training:
          return None, bottom_outputs, None
        else:
          # For inference, we want to return the full image with generated bottom half
          # Concatenate the original top half with the generated bottom half
          
          full_image = torch.cat([original_top_half, bottom_outputs], dim=2)
          return full_image, None, None
