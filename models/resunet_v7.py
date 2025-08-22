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




def fixed_noise_level():
    """Return a fixed noise level"""
    return 0.1

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
  

      
class ResUNet384V7(nn.Module):
    def __init__(self):
        super(ResUNet384V7, self).__init__()
        
        self.noise_level = fixed_noise_level()

        self.ellipse_params = {
            'center': (0.0, 0),  # (x,y)中心偏移(归一化坐标)
            'axes': (1, 0.75),      # (宽,高)比例
            'blur': 0.0             # 边缘模糊系数(相对于短边)
        }
        
        # --- First UNet (processes bottom half) ---
        self.bottom_unet_encoder1 = construct_encoder_layers(6, 3, 16, 1, kernel=3, add_spatial=True)
        self.bottom_unet_down1 = construct_encoder_layers(6, 16, 16, 2, add_spatial=True)
        
        self.bottom_unet_encoder2 = construct_encoder_layers(6, 16, 32, 1, add_spatial=True)
        self.bottom_unet_down2 = construct_encoder_layers(6, 32, 32, 2, add_spatial=True)
               
        self.bottom_unet_encoder3 = construct_encoder_layers(3, 32, 64, 1, add_spatial=True)
        self.bottom_unet_down3 = construct_encoder_layers(3, 64, 64, 2, add_spatial=True)

        self.bottom_unet_encoder4 = construct_encoder_layers(2, 64, 128, 1, add_spatial=True)
        self.bottom_unet_pos_encoder4 = LearnablePositionalEncoding2D(d_model=128, max_h=24, max_w=48, dropout=0.1)
        self.bottom_unet_down4 = construct_encoder_layers(2, 128, 128, 2)
        
        self.bottom_unet_encoder5 = construct_encoder_layers(2, 128, 256, 1, add_spatial=True)
        self.bottom_unet_pos_encoder5 = LearnablePositionalEncoding2D(d_model=256, max_h=12, max_w=24, dropout=0.1)
        self.bottom_unet_down5 = construct_encoder_layers(2, 256, 256, 2)

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
            
            Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
        )
        self.audio_adapter1 = nn.AdaptiveAvgPool2d((48, 48)) # Target size for cross-attention
        self.audio_pos_encoder_ca = LearnablePositionalEncoding2D(d_model=256, max_h=48, max_w=48, dropout=0.1)

        self.bottom_unet_bottleneck = construct_encoder_layers(3, 256, 256, 1, True)
        self.bottom_unet_bottleneck_pos_encoder = LearnablePositionalEncoding2D(d_model=256, max_h=6, max_w=12, dropout=0.1)
        
        # Audio adapters for fusing with bottom decoders
        self.audio_adapter_bottleneck = nn.AdaptiveAvgPool2d((6, 12))  # Match bottleneck spatial dims
        self.audio_adapter_de5 = nn.AdaptiveAvgPool2d((12, 24))       # Match de5 spatial dims
        self.audio_adapter_de4 = nn.AdaptiveAvgPool2d((24, 48))       # Match de4 spatial dims
        
        # Audio adapters for fusing with final decoders
        self.audio_adapter_final_bottleneck = nn.AdaptiveAvgPool2d((12, 12))  # Match final bottleneck spatial dims
        self.audio_adapter_final_de5 = nn.AdaptiveAvgPool2d((24, 24))        # Match final de5 spatial dims
        self.audio_adapter_final_de4 = nn.AdaptiveAvgPool2d((48, 48))        # Match final de4 spatial dims
        
        # Bottom UNet Decoders
        self.bottom_unet_decoder5 = construct_decoder_layers(2, 512, 128, 2)  # 256 (bottom_bottleneck) + 256 (audio) = 512
        self.bottom_unet_conv5 = construct_encoder_layers(2, 640, 320, 1) # 128 (debottom5) + 256 (bottom5) = 384
        
        self.bottom_unet_decoder4 = construct_decoder_layers(3, 320, 160, 2)  # 128 (bottom_de5) + 256 (audio) = 384
        self.bottom_unet_conv4 = construct_encoder_layers(3, 544, 128, 1) # 128 (debottom4) + 256 (bottom4) = 384
        
        self.bottom_unet_decoder3 = construct_decoder_layers(3, 128, 64, 2)
        self.bottom_unet_conv3 = construct_encoder_layers(3, 128, 64, 1) # 64 (debottom3) + 128 (bottom3) = 192
        
        self.bottom_unet_decoder2 = construct_decoder_layers(6, 64, 32, 2, add_spatial=True)
        self.bottom_unet_conv2 = construct_encoder_layers(6, 64, 32, 1, add_spatial=True) # 32 (debottom2) + 64 (bottom2) = 96

        self.bottom_unet_decoder1 = construct_decoder_layers(6, 32, 32, 2, add_spatial=True)
        self.bottom_unet_decoder0 = construct_encoder_layers(6, 32, 32, 1, add_spatial=True)
        
        self.bottom_unet_conv1 = construct_encoder_layers(3, 48, 32, 1) # 32 (debottom1 from prev_decoder0) + 64 (bottom1) = 96

        self.bottom_unet_output_block = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        
        # --- Second UNet (processes combined full image + reference) ---
        self.final_unet_encoder1 = construct_encoder_layers(6, 12, 24, 1, kernel=3, add_spatial=True) # 3 (combined) + 9 (ref) = 12
        self.final_unet_down1 = construct_encoder_layers(6, 24, 24, 2, add_spatial=True)
        
        self.final_unet_encoder2 = construct_encoder_layers(6, 24, 48, 1, add_spatial=True)
        self.final_unet_down2 = construct_encoder_layers(6, 48, 48, 2, add_spatial=True)
               
        self.final_unet_encoder3 = construct_encoder_layers(3, 48, 96, 1, add_spatial=True)
        self.final_unet_down3 = construct_encoder_layers(3, 96, 96, 2, add_spatial=True)

        self.final_unet_encoder4 = construct_encoder_layers(2, 96, 192, 1, add_spatial=True)
        self.final_unet_pos_encoder4 = LearnablePositionalEncoding2D(d_model=192, max_h=48, max_w=48, dropout=0.1)
        self.final_unet_down4 = construct_encoder_layers(2, 192, 192, 2)
        
        self.final_unet_encoder5 = construct_encoder_layers(2, 192, 192, 1, add_spatial=True)
        self.final_unet_pos_encoder5 = LearnablePositionalEncoding2D(d_model=192, max_h=24, max_w=24, dropout=0.1)
        self.final_unet_down5 = construct_encoder_layers(2, 192, 192, 2)

        self.final_unet_bottleneck = construct_encoder_layers(3, 192, 192, 1, True)  # 192 (final_down5) + 256 (audio) = 448
        self.final_unet_bottleneck_pos_encoder = LearnablePositionalEncoding2D(d_model=192, max_h=12, max_w=12, dropout=0.1)
        
        # Final UNet Decoders
        self.final_unet_decoder5 = construct_decoder_layers(2, 448, 224, 2)  # 192 (final_bottleneck) + 256 (audio) = 448
        self.final_unet_conv5 = construct_encoder_layers(2, 672, 336, 1) # 96 (definal5) + 192 (final5) = 288
        
        self.final_unet_decoder4 = construct_decoder_layers(3, 336, 168, 2)  # 96 (final_de5) + 256 (audio) = 352
        self.final_unet_conv4 = construct_encoder_layers(3, 616, 128, 1) # 128 (definal4) + 512 (final4) = 640
        
        self.final_unet_decoder3 = construct_decoder_layers(3, 128, 64, 2)
        self.final_unet_conv3 = construct_encoder_layers(3, 160, 64, 1) # 64 (definal3) + 256 (final3) = 320
        
        self.final_unet_decoder2 = construct_decoder_layers(6, 64, 32, 2, add_spatial=True)
        self.final_unet_conv2 = construct_encoder_layers(6, 80, 32, 1, add_spatial=True) # 32 (definal2) + 128 (final2) = 160

        self.final_unet_decoder1 = construct_decoder_layers(6, 32, 32, 2, add_spatial=True)
        self.final_unet_decoder0 = construct_encoder_layers(6, 32, 32, 1, add_spatial=True)
        
        self.final_unet_conv1 = construct_encoder_layers(3, 56, 32, 1) # 32 (definal1 from prev_decoder0) + 64 (final1) = 96

        self.final_unet_output_block = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        
        
        
    
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
        
        # Use fixed noise level instead of diffusion steps
        noise = torch.randn_like(rgb_channels)
        noisy_rgb = math.sqrt(1 - self.noise_level) * rgb_channels + math.sqrt(self.noise_level) * noise
        
        noisy_rgb = rgb_channels * (1 - mask) + noisy_rgb * mask
        
        noisy_bottom = torch.cat([noisy_rgb, other_channels], dim=1)
        result = torch.cat([top_half, noisy_bottom], dim=2)

        return result
                
    def forward(self, audio_sequences, face_sequences, step=None, train_face_enhancer=True):
        input_dim_size = len(face_sequences.size())
        B = audio_sequences.size(0)       
        
        if input_dim_size > 4:
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)

        # Process audio sequences
        audio_embedding = self.audio_encoder1(audio_sequences)
        audio_embedding = self.audio_adapter1(audio_embedding)  # Adapt to (B, 256, 48, 48)
        audio_embedding = self.audio_pos_encoder_ca(audio_embedding)
        
        # Extract first 3 channels for processing
        face_sequences_3ch = face_sequences[:, :3, :, :]
        
        # Apply diffusion to the face sequences
        face_sequences_3ch = self.diffuse(face_sequences_3ch, channels_to_mask=3)
        
        # Extract remaining 9 channels as reference
        face_sequences_ref = face_sequences[:, 3:, :, :]
        
        # Get image dimensions
        _, _, h, w = face_sequences_3ch.shape
        split_idx = h // 2
        
        # Split into top and bottom halves
        top_half = face_sequences_3ch[:, :, :split_idx, :]
        bottom_half = face_sequences_3ch[:, :, split_idx:, :]
        
        # First UNet: Process bottom half to generate bottom half output
        # Encode bottom half through bottom UNet
        bottom_enc1 = self.bottom_unet_encoder1(bottom_half)
        bottom_down1 = self.bottom_unet_down1(bottom_enc1)
        
        bottom_enc2 = self.bottom_unet_encoder2(bottom_down1)
        bottom_down2 = self.bottom_unet_down2(bottom_enc2)
        
        bottom_enc3 = self.bottom_unet_encoder3(bottom_down2)
        bottom_down3 = self.bottom_unet_down3(bottom_enc3)
        
        bottom_enc4 = self.bottom_unet_encoder4(bottom_down3)
        bottom_enc4 = self.bottom_unet_pos_encoder4(bottom_enc4)
        bottom_down4 = self.bottom_unet_down4(bottom_enc4)
        
        bottom_enc5 = self.bottom_unet_encoder5(bottom_down4)
        bottom_enc5 = self.bottom_unet_pos_encoder5(bottom_enc5)
        bottom_down5 = self.bottom_unet_down5(bottom_enc5)
        
        bottom_bottleneck = self.bottom_unet_bottleneck(bottom_down5)
        bottom_bottleneck = self.bottom_unet_bottleneck_pos_encoder(bottom_bottleneck)
        
        # Fuse audio with bottom bottleneck
        audio_bottleneck = self.audio_adapter_bottleneck(audio_embedding)
        bottom_bottleneck = torch.cat([bottom_bottleneck, audio_bottleneck], dim=1)
        
        # Decode to generate bottom half output
        bottom_de5 = self.bottom_unet_decoder5(bottom_bottleneck)
        # Fuse audio with bottom de5
        audio_de5 = self.audio_adapter_de5(audio_embedding)
        bottom_de5 = torch.cat([bottom_de5, audio_de5], dim=1)
        bottom_cat5 = torch.cat([bottom_de5, bottom_enc5], dim=1)
        bottom_cat5 = self.bottom_unet_conv5(bottom_cat5)
        
        bottom_de4 = self.bottom_unet_decoder4(bottom_cat5)
        # Fuse audio with bottom de4
        audio_de4 = self.audio_adapter_de4(audio_embedding)
        bottom_de4 = torch.cat([bottom_de4, audio_de4], dim=1)
        bottom_cat4 = torch.cat([bottom_de4, bottom_enc4], dim=1)
        bottom_cat4 = self.bottom_unet_conv4(bottom_cat4)
        
        bottom_de3 = self.bottom_unet_decoder3(bottom_cat4)
        bottom_cat3 = torch.cat([bottom_de3, bottom_enc3], dim=1)
        bottom_cat3 = self.bottom_unet_conv3(bottom_cat3)
        
        bottom_de2 = self.bottom_unet_decoder2(bottom_cat3)
        bottom_cat2 = torch.cat([bottom_de2, bottom_enc2], dim=1)
        bottom_cat2 = self.bottom_unet_conv2(bottom_cat2)
        
        bottom_de1 = self.bottom_unet_decoder1(bottom_cat2)
        bottom_de0 = self.bottom_unet_decoder0(bottom_de1)
        bottom_cat1 = torch.cat([bottom_de0, bottom_enc1], dim=1)
        bottom_cat1 = self.bottom_unet_conv1(bottom_cat1)
        
        generated_bottom_half = self.bottom_unet_output_block(bottom_cat1)
        
        # Combine top half with generated bottom half to form full image
        combined_full_image = torch.cat([top_half, generated_bottom_half], dim=2)
        
        # Concatenate combined full image with reference via channel dimension
        combined_with_ref = torch.cat([combined_full_image, face_sequences_ref], dim=1)
        
        # Second UNet: Process combined result to generate final output
        # Encode combined result through final UNet
        final_enc1 = self.final_unet_encoder1(combined_with_ref)
        final_down1 = self.final_unet_down1(final_enc1)
        
        final_enc2 = self.final_unet_encoder2(final_down1)
        final_down2 = self.final_unet_down2(final_enc2)
        
        final_enc3 = self.final_unet_encoder3(final_down2)
        final_down3 = self.final_unet_down3(final_enc3)
        
        final_enc4 = self.final_unet_encoder4(final_down3)
        final_enc4 = self.final_unet_pos_encoder4(final_enc4)
        final_down4 = self.final_unet_down4(final_enc4)
        
        final_enc5 = self.final_unet_encoder5(final_down4)
        final_enc5 = self.final_unet_pos_encoder5(final_enc5)
        final_down5 = self.final_unet_down5(final_enc5)
        
        final_bottleneck = self.final_unet_bottleneck(final_down5)
        final_bottleneck = self.final_unet_bottleneck_pos_encoder(final_bottleneck)
        
        # Fuse audio with final bottleneck
        audio_final_bottleneck = self.audio_adapter_final_bottleneck(audio_embedding)
        final_bottleneck = torch.cat([final_bottleneck, audio_final_bottleneck], dim=1)
        
        # Decode to generate final output
        final_de5 = self.final_unet_decoder5(final_bottleneck)
        # Fuse audio with final de5
        audio_final_de5 = self.audio_adapter_final_de5(audio_embedding)
        final_de5 = torch.cat([final_de5, audio_final_de5], dim=1)
        final_cat5 = torch.cat([final_de5, final_enc5], dim=1)
        final_cat5 = self.final_unet_conv5(final_cat5)
        
        final_de4 = self.final_unet_decoder4(final_cat5)
        # Fuse audio with final de4
        audio_final_de4 = self.audio_adapter_final_de4(audio_embedding)
        final_de4 = torch.cat([final_de4, audio_final_de4], dim=1)
        final_cat4 = torch.cat([final_de4, final_enc4], dim=1)
        final_cat4 = self.final_unet_conv4(final_cat4)
        
        final_de3 = self.final_unet_decoder3(final_cat4)
        final_cat3 = torch.cat([final_de3, final_enc3], dim=1)
        final_cat3 = self.final_unet_conv3(final_cat3)
        
        final_de2 = self.final_unet_decoder2(final_cat3)
        final_cat2 = torch.cat([final_de2, final_enc2], dim=1)
        final_cat2 = self.final_unet_conv2(final_cat2)
        
        final_de1 = self.final_unet_decoder1(final_cat2)
        final_de0 = self.final_unet_decoder0(final_de1)
        final_cat1 = torch.cat([final_de0, final_enc1], dim=1)
        final_cat1 = self.final_unet_conv1(final_cat1)
        
        final_output = self.final_unet_output_block(final_cat1)
        
        if input_dim_size > 4:
            outputs = torch.split(final_output, B, dim=0)
            outputs = torch.stack(outputs, dim=2)
        else:
            outputs = final_output
        
        
        return outputs, None, None
