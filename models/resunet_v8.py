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


class SimpleAudioFusion(nn.Module):
    """Simpler fusion that just adds projected audio features to visual features"""
    def __init__(self, visual_dim, audio_dim):
        super().__init__()
        self.audio_proj = nn.Conv2d(audio_dim, visual_dim, 1)
        
    def forward(self, visual_feat, audio_feat):
        # Resize audio to match visual spatial dimensions
        if audio_feat.size(2) != visual_feat.size(2) or audio_feat.size(3) != visual_feat.size(3):
            audio_feat = F.interpolate(audio_feat, size=(visual_feat.size(2), visual_feat.size(3)), 
                                     mode='bilinear', align_corners=False)
        
        audio_feat = self.audio_proj(audio_feat)
        return visual_feat + audio_feat

      
class ResUNet384V8(nn.Module):
    def __init__(self):
        super(ResUNet384V8, self).__init__()
        
        self.ellipse_params = {
            'center': (0.0, 0),  # (x,y)中心偏移(归一化坐标)
            'axes': (1, 0.75),      # (宽,高)比例
            'blur': 0.0             # 边缘模糊系数(相对于短边)
        }
        
        # --- First UNet (processes bottom half) ---
        self.bottom_unet_encoder1 = construct_encoder_layers(3, 6, 24, 1)
        self.bottom_unet_down1 = construct_encoder_layers(3, 24, 48, 2)
                
        self.bottom_unet_encoder2 = construct_encoder_layers(3, 48, 96, 1)
        self.bottom_unet_down2 = construct_encoder_layers(3, 352, 96, 2)
                       
        self.bottom_unet_encoder3 = construct_encoder_layers(3, 96, 192, 1)
        self.bottom_unet_down3 = construct_encoder_layers(3, 448, 192, 2)

        self.bottom_unet_encoder4 = construct_encoder_layers(2, 192, 192, 1)
        self.bottom_unet_pos_encoder4 = LearnablePositionalEncoding2D(d_model=192, max_h=24, max_w=48, dropout=0.1)
        self.bottom_unet_down4 = construct_encoder_layers(2, 448, 192, 2)
        
        self.bottom_unet_encoder5 = construct_encoder_layers(2, 192, 192, 1)
        self.bottom_unet_pos_encoder5 = LearnablePositionalEncoding2D(d_model=192, max_h=12, max_w=24, dropout=0.1)
        self.bottom_unet_down5 = construct_encoder_layers(2, 448, 192, 2)

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
            
            Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            
            Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
        )
                

        self.bottom_unet_bottleneck = construct_encoder_layers(3, 192, 256, 1)
        self.bottom_unet_bottleneck_pos_encoder = LearnablePositionalEncoding2D(d_model=256, max_h=6, max_w=12, dropout=0.1)
        
        # Audio adapters for fusing with bottom decoders at multiple levels
        self.audio_adapter_bottleneck = nn.AdaptiveAvgPool2d((6, 12))  # Match bottleneck spatial dims
        self.audio_adapter_5 = nn.AdaptiveAvgPool2d((12, 24))   # Match decoder4 spatial dims
        self.audio_adapter_4 = nn.AdaptiveAvgPool2d((24, 48))   # Match decoder4 spatial dims
        self.audio_adapter_3 = nn.AdaptiveAvgPool2d((48, 96))   # Match decoder3 spatial dims  
        self.audio_adapter_2 = nn.AdaptiveAvgPool2d((96, 192))   # Match decoder2 spatial dims
        
        # Positional encoders for audio at different scales
        self.audio_pos_encoder_bottleneck = LearnablePositionalEncoding2D(d_model=256, max_h=6, max_w=12, dropout=0.1)
        self.audio_pos_encoder_5 = LearnablePositionalEncoding2D(d_model=256, max_h=12, max_w=24, dropout=0.1)
        self.audio_pos_encoder_4 = LearnablePositionalEncoding2D(d_model=256, max_h=24, max_w=48, dropout=0.1)
        self.audio_pos_encoder_3 = LearnablePositionalEncoding2D(d_model=256, max_h=48, max_w=96, dropout=0.1)
        self.audio_pos_encoder_2 = LearnablePositionalEncoding2D(d_model=256, max_h=96, max_w=192, dropout=0.1)
       
                
        # Bottom UNet Decoders
        self.bottom_unet_decoder5 = construct_decoder_layers(3, 512, 256, 2)  # 256 (bottom_bottleneck) + 256 (audio) = 512
        self.bottom_unet_conv5 = construct_encoder_layers(3, 704, 320, 1) # 128 (debottom5) + 256 (bottom5) = 384
        
        self.bottom_unet_decoder4 = construct_decoder_layers(3, 320, 256, 2)  # 128 (bottom_de5) + 256 (audio) = 384
        self.bottom_unet_conv4 = construct_encoder_layers(3, 704, 256, 1) # 128 (debottom4) + 256 (bottom4) = 384
        
        self.bottom_unet_decoder3 = construct_decoder_layers(3, 256, 256, 2)
        self.bottom_unet_conv3 = construct_encoder_layers(3, 704, 256, 1) # 64 (debottom3) + 128 (bottom3) = 192
        
        self.bottom_unet_decoder2 = construct_decoder_layers(3, 256, 128, 2, add_spatial=True)
        self.bottom_unet_conv2 = construct_encoder_layers(3, 480, 128, 1) # 32 (debottom2) + 64 (bottom2) = 96

        self.bottom_unet_decoder1 = construct_decoder_layers(3, 128, 64, 2, add_spatial=True)
        self.bottom_unet_decoder0 = construct_encoder_layers(3, 64, 32, 1)
                
        self.bottom_unet_conv1 = construct_encoder_layers(3, 56, 32, 1) # 32 (debottom1 from prev_decoder0) + 64 (bottom1) = 96

        self.bottom_unet_output_block = nn.Sequential(
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
        audio_5 = self.audio_pos_encoder_5(audio_5)
        
        audio_4 = self.audio_adapter_4(audio_embedding)
        audio_4 = self.audio_pos_encoder_4(audio_4)
        
        audio_3 = self.audio_adapter_3(audio_embedding)
        audio_3 = self.audio_pos_encoder_3(audio_3)
        
        audio_2 = self.audio_adapter_2(audio_embedding)
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
        bottom_down1 = self.bottom_unet_down1(bottom_enc1)
        
        bottom_enc2 = self.bottom_unet_encoder2(bottom_down1)
        bottom_enc2 = torch.cat([bottom_enc2, audio_2], dim=1)
        bottom_down2 = self.bottom_unet_down2(bottom_enc2)

                
        bottom_enc3 = self.bottom_unet_encoder3(bottom_down2)
        bottom_enc3 = torch.cat([bottom_enc3, audio_3], dim=1)
        bottom_down3 = self.bottom_unet_down3(bottom_enc3)
        
        bottom_enc4 = self.bottom_unet_encoder4(bottom_down3)
        bottom_enc4 = torch.cat([bottom_enc4, audio_4], dim=1)
        bottom_down4 = self.bottom_unet_down4(bottom_enc4)
        
        bottom_enc5 = self.bottom_unet_encoder5(bottom_down4)
        bottom_enc5 = torch.cat([bottom_enc5, audio_5], dim=1)
        bottom_down5 = self.bottom_unet_down5(bottom_enc5)
        
        bottom_bottleneck = self.bottom_unet_bottleneck(bottom_down5)
        bottom_bottleneck = self.bottom_unet_bottleneck_pos_encoder(bottom_bottleneck)
        
        
                
        # Fuse audio with bottom bottleneck (original concatenation)
        bottom_bottleneck = torch.cat([bottom_bottleneck, audio_bottleneck], dim=1)
        
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
        combined_full_image = torch.cat([top_half, generated_bottom_half], dim=2)
        
        if input_dim_size > 4:            
            bottom_outputs = torch.split(generated_bottom_half, B, dim=0)
            bottom_outputs = torch.stack(bottom_outputs, dim=2)
            combined_full_image = torch.split(combined_full_image, B, dim=0)
            combined_full_image = torch.stack(combined_full_image, dim=2)
        else:
            bottom_outputs = generated_bottom_half
            
        
        # Save generated bottom half and final output every 1000 steps
        if step is not None and step % 5000 == 0:
            # Create directory for saving generated images
            save_dir = "generated_images"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                
            # Convert generated bottom half to numpy array and save as image
            # Take the first sample in the batch for saving
            bottom_half_to_save = generated_bottom_half[0].detach().cpu().numpy()
            # Transpose from (C, H, W) to (H, W, C) and convert to uint8
            bottom_half_to_save = np.transpose(bottom_half_to_save, (1, 2, 0))
            bottom_half_to_save = (bottom_half_to_save * 255).astype(np.uint8)
            
            # Save the bottom half image
            save_path = os.path.join(save_dir, f"generated_bottom_half_step_{step}.jpg")
            cv2.imwrite(save_path, bottom_half_to_save)
        
        return None, bottom_outputs, None