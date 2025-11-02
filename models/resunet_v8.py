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
        self.bottom_unet_encoder1 = construct_encoder_layers(4, 6, 64, 1)
        self.bottom_unet_down1 = construct_encoder_layers(4, 64, 96, 2)
                
        self.bottom_unet_encoder2 = construct_encoder_layers(4, 96, 128, 1)
        self.bottom_unet_down2 = construct_encoder_layers(4, 128, 192, 2)
                       
        self.bottom_unet_encoder3 = construct_encoder_layers(3, 192, 256, 1)
        self.bottom_unet_down3 = construct_encoder_layers(3, 256, 320, 2)

        self.bottom_unet_encoder4 = construct_encoder_layers(4, 320, 192, 1)
        self.bottom_unet_down4 = construct_encoder_layers(4, 192, 192, 2)
        self.bottom_unet_pos_encoder4 = LearnablePositionalEncoding2D(d_model=192, max_h=12, max_w=24, dropout=0.1)
        
        self.bottom_unet_encoder5 = construct_encoder_layers(2, 192, 192, 1)
        self.bottom_unet_down5 = construct_encoder_layers(2, 192, 192, 2)
        
        # --- MaxPooling for residual connection from encoder1 to encoder3 ---
        self.encoder1_to_encoder2_skip = nn.Sequential(nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
                                                       nn.BatchNorm2d(96),
                                                       nn.LeakyReLU(0.01)
        )
        
        self.encoder1_to_encoder_down2_skip = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                                                       nn.BatchNorm2d(128),
                                                       nn.LeakyReLU(0.01)
        )
        
        self.encoder2_to_encoder3_skip = nn.Sequential(nn.Conv2d(128, 192, kernel_size=3, stride=2, padding=1),
                                                       nn.BatchNorm2d(192),
                                                       nn.LeakyReLU(0.01)
        )
        
        self.encoder3_to_encoder4_skip = nn.Sequential(nn.Conv2d(256, 320, kernel_size=3, stride=2, padding=1),
                                                       nn.BatchNorm2d(320),
                                                       nn.LeakyReLU(0.01)
        )
        
        self.encoder4_to_encoder5_skip = nn.Sequential(nn.Conv2d(192, 192, kernel_size=3, stride=2, padding=1),
                                                       nn.BatchNorm2d(192),
                                                       nn.LeakyReLU(0.01)
        )
        
        
        self.encoder1_to_encoder4_skip = nn.Sequential(
          nn.AdaptiveAvgPool2d((24, 48)),
          nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(192),
          nn.LeakyReLU(0.01)
        )
        

        # --- Audio encoder ---
        self.audio_encoder1 = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1, use_zero_sensitive=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True, use_zero_sensitive=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True, use_zero_sensitive=True),
        )
        
        # Add maxpool to match audio_encoder2 output shape
        self.audio1_pool = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.audio1_conv = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        self.audio_encoder2 = nn.Sequential(
            Conv2d(32, 64, kernel_size=3, stride=(2, 1), padding=1, use_zero_sensitive=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True, use_zero_sensitive=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True, use_zero_sensitive=True),
        )
        
        self.audio2_pool = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.audio2_conv = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        
        self.audio_encoder3 = nn.Sequential(
            Conv2d(64, 128, kernel_size=3, stride=1, padding=1, use_zero_sensitive=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True, use_zero_sensitive=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True, use_zero_sensitive=True),
        )
        
        self.audio3_conv = nn.Conv2d(128, 192, kernel_size=3)
        self.audio3_conv_decoder = nn.Conv2d(128, 448, kernel_size=3, stride=1, padding=1)

        
        self.audio_encoder4 = nn.Sequential(
            Conv2d(128, 192, kernel_size=3, stride=2, padding=1, use_zero_sensitive=True),
            Conv2d(192, 192, kernel_size=3, stride=1, padding=1, residual=True, use_zero_sensitive=True),
        )
        
        
        # Audio adapters for fusing with bottom decoders at multiple levels
        self.audio_adapter_3 = nn.AdaptiveAvgPool2d((24, 48))   # Match decoder5 spatial dims
        self.audio_adapter_3_conv = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                                                       nn.BatchNorm2d(256),
                                                       nn.LeakyReLU(0.01))
        
                
        self.audio_adapter_4 = nn.AdaptiveAvgPool2d((12, 24))   # Match decoder5 spatial dims
        self.audio_adapter_4_conv = nn.Identity()
        
                
        # Positional encoders for audio at different scales
        self.audio_pos_encoder_4 = LearnablePositionalEncoding2D(d_model=192, max_h=12, max_w=24, dropout=0.1)
        self.audio_pos_encoder_decoder4 = LearnablePositionalEncoding2D(d_model=192, max_h=24, max_w=48, dropout=0.1)
        self.audio_pos_encoder_decoder3 = LearnablePositionalEncoding2D(d_model=256, max_h=48, max_w=96, dropout=0.1)
        
                
        # Bottom UNet Decoders
        # Updated decoder connections after removing encoder5/down5
        self.bottom_unet_decoder5 = construct_decoder_layers(2, 192, 192, 2)
        self.bottom_unet_conv5 = construct_encoder_layers(2, 192, 192, 1) # 256 (debottom4) + 192 (bottom4) = 384
        
        self.bottom_unet_decoder4 = construct_decoder_layers(4, 384, 256, 2)
        self.bottom_unet_conv4 = construct_encoder_layers(4, 448, 256, 1) # 256 (debottom4) + 192 (bottom4) = 384
        
        self.bottom_unet_decoder3 = construct_decoder_layers(3, 512, 320, 2)
        self.bottom_unet_conv3 = construct_encoder_layers(3, 576, 256, 1) # 256 (debottom3) + 192 (bottom3) = 448
        
        self.bottom_unet_decoder2 = construct_decoder_layers(4, 256, 128, 2)
        self.bottom_unet_conv2 = construct_encoder_layers(4, 256, 128, 1) # 128 (debottom2) + 96 (bottom2) = 224

        self.bottom_unet_decoder1 = construct_decoder_layers(4, 128, 64, 2)
        self.bottom_unet_decoder0 = construct_encoder_layers(4, 128, 64, 1)

        self.bottom_unet_output_block = nn.Sequential(
            Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        
        # Transformer Encoder
        self.face4_cross_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=384, nhead=4, dropout=0.1, activation='gelu', batch_first=True),
            num_layers=2
        )
        
        self.face4_attn_reduce = nn.Conv2d(384, 192, kernel_size=3, stride=1, padding=1)
                
        self.face_norm = nn.BatchNorm2d(192)  # For face features
        self.audio_norm = nn.BatchNorm2d(192)  # For audio features
                
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
        
        masked_rgb = rgb_channels * (1 - mask)  # 非遮罩区域保留原值

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
        
        
        # Extract first 3 channels for processing
        face_sequences_3ch = face_sequences[:, :3, :, :]
        
        # Apply diffusion to the face sequences
        face_sequences_3ch = self.diffuse(face_sequences_3ch, channels_to_mask=3)
        
        # Extract remaining channels as reference
        face_sequences_ref = face_sequences[:, 3:, :, :]
        
        # Get image dimensions
        batch_size, _, h, w = face_sequences_3ch.shape
        split_idx = h // 2
        
        # Extract 9 reference channels for bottom encoder
        bottom_ref_channels = face_sequences_ref[:, :, split_idx:, :]
        
        # Split into top and bottom halves
        bottom_half = face_sequences_3ch[:, :, split_idx:, :]
        
        
        # Combine bottom half with reference channels for bottom encoder
        bottom_half_with_ref = torch.cat([bottom_half, bottom_ref_channels], dim=1)
        
        
        # Process audio sequences
        audio_1 = self.audio_encoder1(audio_sequences)
        audio1_to_audio3 = self.audio1_pool(audio_1)
        audio1_to_audio3 = self.audio1_conv(audio1_to_audio3)
        
                
        audio_2 = self.audio_encoder2(audio_1)
        #audio2_to_audio4 = self.audio2_pool(audio_2)
        audio2_to_audio4 = self.audio2_conv(audio_2)
                
        audio_3 = self.audio_encoder3(audio_2 + audio1_to_audio3)       
        audio3_adapted = self.audio_adapter_3(audio_3)
        audio3_adapted = self.audio_adapter_3_conv(audio3_adapted)
        audio3_adapted = self.audio_pos_encoder_decoder3(audio3_adapted)
        
        audio_4 = self.audio_encoder4(audio_3 + audio2_to_audio4)        
        audio4_adapted = self.audio_adapter_4(audio_4)
        audio4_adapted = self.audio_adapter_4_conv(audio4_adapted)
        audio4_adapted = self.audio_pos_encoder_4(audio4_adapted)
                        
        # First UNet: Process bottom half to generate bottom half output
        # Encode bottom half through bottom UNet
        bottom_enc1 = self.bottom_unet_encoder1(bottom_half_with_ref) #192x384
        encoder1_to_encoder2_residual = self.encoder1_to_encoder2_skip(bottom_enc1)  # Downsample spatially 96x192
        
        encoder1_to_encoder_down2_residual = self.encoder1_to_encoder_down2_skip(bottom_enc1) # 96x192
        
                
        encoder1_to_encoder4_residual = self.encoder1_to_encoder4_skip(bottom_enc1)  # Downsample spatially
                        
        bottom_down1 = self.bottom_unet_down1(bottom_enc1)
        
        
        bottom_enc2 = self.bottom_unet_encoder2(bottom_down1 + encoder1_to_encoder2_residual)
        
        encoder2_to_encoder3_skip = self.encoder2_to_encoder3_skip(bottom_enc2)
                
        bottom_down2 = self.bottom_unet_down2(bottom_enc2 + encoder1_to_encoder_down2_residual)
        
        bottom_enc3 = self.bottom_unet_encoder3(bottom_down2 + encoder2_to_encoder3_skip)
        
        encoder3_to_encoder4_skip = self.encoder3_to_encoder4_skip(bottom_enc3)
                
        bottom_down3 = self.bottom_unet_down3(bottom_enc3)
        
        bottom_enc4 = self.bottom_unet_encoder4(bottom_down3 + encoder3_to_encoder4_skip)
        
        encoder4_to_encoder5_skip = self.encoder4_to_encoder5_skip(bottom_enc4)
        
        
        bottom_down4 = self.bottom_unet_down4(bottom_enc4 + encoder1_to_encoder4_residual)
        bottom_down4 = self.bottom_unet_pos_encoder4(bottom_down4 + encoder4_to_encoder5_skip)
                
        _, channels, height, width = bottom_down4.shape
        # Concatenate along channel dimension

        flat_face4_down = self.face_norm(bottom_down4)
        flat_face4_down = flat_face4_down.permute(0, 2, 3, 1)  # [B, H, W, C]
        flat_face4_down = flat_face4_down.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)  # [B, seq_len, features]

        # For audio features  
        flat_audio5 = self.audio_norm(audio4_adapted)
        flat_audio4 = flat_audio5.permute(0, 2, 3, 1)  # [B, H, W, C]
        
        flat_audio4 = flat_audio4.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)  # [B, seq_len, features]

        # Concatenate along sequence dimension
        face4_combined_sequence = torch.cat([flat_face4_down, flat_audio4], dim=2)  # [10, 576, 192]
                
        # Process through transformer
        face4_fused = self.face4_cross_attn(face4_combined_sequence)  # [10, 288, 384]
        face4_fused = face4_fused + face4_combined_sequence
        face4_fused = face4_fused.permute(0, 2, 1)  # [10, 192, 288]
        #face4_fused = self.face4_proj(face4_fused)
        face4_fused = face4_fused.view(batch_size, 384, height, width)  # [10, 192, 12, 24]
        face4_fused = self.face4_attn_reduce(face4_fused)                               
        face4_fused = bottom_down4 + face4_fused
        
        bottom_enc5 = self.bottom_unet_encoder5(bottom_down4 + face4_fused)
        bottom_down5 = self.bottom_unet_down5(bottom_enc5)
        
        bottom_dec5 = self.bottom_unet_decoder5(bottom_down5)
        bottom_dec5_up = self.bottom_unet_conv5(bottom_dec5)
        
        
        # Decode to generate bottom half output with multi-level audio fusion
        fusion = torch.cat([bottom_dec5_up, audio4_adapted + face4_fused], dim=1)
        bottom_de4 = self.bottom_unet_decoder4(fusion)
        bottom_cat4 = torch.cat([bottom_de4, bottom_enc4], dim=1)
        bottom_cat4 = self.bottom_unet_conv4(bottom_cat4)
        
        fused_for_decoder3 = torch.cat([bottom_cat4, audio3_adapted], dim=1)
        bottom_de3 = self.bottom_unet_decoder3(fused_for_decoder3)
        bottom_cat3 = torch.cat([bottom_de3, bottom_enc3], dim=1)
        bottom_cat3 = self.bottom_unet_conv3(bottom_cat3)
        
        bottom_de2 = self.bottom_unet_decoder2(bottom_cat3)
        bottom_cat2 = torch.cat([bottom_de2, bottom_enc2], dim=1)
        bottom_cat2 = self.bottom_unet_conv2(bottom_cat2)
        
        bottom_de1 = self.bottom_unet_decoder1(bottom_cat2)
        bottom_cat1 = torch.cat([bottom_de1, bottom_enc1], dim=1)
        bottom_de0 = self.bottom_unet_decoder0(bottom_cat1)
        
        generated_bottom_half = self.bottom_unet_output_block(bottom_de0)
        
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
