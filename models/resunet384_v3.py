import torch
import math
from torch import nn, random
from torch.nn import functional as F

from .conv import Conv2dTranspose, Conv2d
from .transformer_syncnet import LearnablePositionalEncoding2D
import torch
from torchvision.utils import save_image
import os
from datetime import datetime
import os, random, cv2, argparse
    
def cosine_noise_schedule(num_steps, s=0.008):

    #num_steps = num_steps - 1 # make it 1 step less, and append a no noice step manually at the end
    steps = torch.linspace(0, num_steps, num_steps + 1)
    # Remove the extra .cos() - it's already cosine!
    f = torch.cos((steps / num_steps + s) / (1 + s) * (math.pi / 2))
    alphas = f ** 2  # Squaring is correct (cos²)
    alphas = alphas / alphas[0]  # Normalize to start at 1
    betas = 1 - (alphas[1:] / alphas[:-1])
    result = betas.clamp(min=0.01, max=0.3)

    return result

def linear_schedule():
    return torch.tensor([0.01, 0.15, 0.2, 0.3])
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_att = torch.cat([avg_out, max_out], dim=1)
        x_att = self.conv1(x_att)
        x_att = self.sigmoid(x_att)
        return x + x_att * x  # 使用skip connection
      
class ResUNet384V3(nn.Module):
    def __init__(self):
        super(ResUNet384V3, self).__init__()
        
        num_diffusion_steps = 2
        self.num_diffusion_steps = num_diffusion_steps
        self.betas = linear_schedule()  # or cosine_noise_schedule()
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
              
        self.face_encoder1 = self.construct_encoder_layers(2, 12, 64, 1)
        self.fe_down1 = self.construct_encoder_layers(2, 64, 64, 2)
        
        
        self.face_encoder1_bottom = self.construct_encoder_layers(2, 12, 64, 1)
        self.fe_down1_bottom = self.construct_encoder_layers(2, 64, 64, 2)
        
        self.face_encoder2 = self.construct_encoder_layers(2, 128, 128, 1, True)
        self.fe_down2 = self.construct_encoder_layers(2, 128, 128, 2)
        
        self.face_encoder2_moe1 = self.construct_encoder_layers(2, 128, 128, 1, True)
        self.fe_down2_moe1 = self.construct_encoder_layers(2, 128, 128, 2)

        self.face_encoder3 = self.construct_encoder_layers(2, 256, 256, 1, True)
        self.fe_down3 = self.construct_encoder_layers(2, 256, 256, 2, True)

        self.face_encoder4 = self.construct_encoder_layers(2, 256, 512, 1, True)
        self.fe_down4 = self.construct_encoder_layers(2, 512, 512, 2)
        
        self.face_encoder5 = self.construct_encoder_layers(2, 512, 512, 1, True)
        self.fe_down5 = self.construct_encoder_layers(2, 512, 512, 2)
        
        self.face_transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8, dropout=0.1, activation='gelu'),
            num_layers=2
        )

        # --- Audio encoder ---
        self.audio_encoder1 = nn.Sequential(
            # Example input shape: (B, 1, H_audio, W_audio)
            Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
       
            Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
       
            Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
       
            Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            nn.AdaptiveAvgPool2d((12, 12))
        )
        
        self.audio_transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8, dropout=0.1, activation='gelu'),
            num_layers=2
        )

        self.bottlenet = self.construct_encoder_layers(1, 1024, 1024, 1, True)
                
        
        # Decoders
        self.face_decoder5 = self.construct_decoder_layers(2, 1024, 512, 2, True)
        self.fd_conv5 = self.construct_encoder_layers(2, 1024, 512, 1, True)
        
        self.face_decoder4 = self.construct_decoder_layers(2, 1024, 512, 2, True)
        self.fd_conv4 = self.construct_encoder_layers(2, 1024, 512, 1, True)
        
        self.face_decoder3 = self.construct_decoder_layers(2, 1024, 256, 2, True)
        self.fd_conv3 = self.construct_encoder_layers(2, 512, 256, 1, True)
        
        self.face_decoder2 = self.construct_decoder_layers(2, 512, 128, 2, True)
        self.fd_conv2 = self.construct_encoder_layers(2, 256, 128, 1, True)

        self.face_decoder1 = self.construct_decoder_layers(2, 256, 64, 2, True)
        self.fd_conv1 = self.construct_encoder_layers(2, 128, 64, 1, True)
        

        self.output_block = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        
        self.face_pos_encoder = LearnablePositionalEncoding2D(d_model=512, max_h=24, max_w=24, dropout=0.1)
        self.audio_pos_encoder = LearnablePositionalEncoding2D(d_model=512, max_h=24, max_w=24, dropout=0.1)
        
    def construct_encoder_layers(self, num_of_layers, input_channels, output_channels, first_layer_stride, add_spatial=False):
        layers = []
        # First layer
        layers.append(Conv2d(input_channels, output_channels, kernel_size=3, stride=first_layer_stride, padding=1))
        # Subsequent layers
        for _ in range(num_of_layers - 1):
            layers.append(Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1, residual=True))
        # Optional SpatialAttention
        if add_spatial:
            layers.append(SpatialAttention())  # Assumes SpatialAttention is a PyTorch Module
        
        return nn.Sequential(*layers)

    def construct_decoder_layers(self, num_of_layers, input_channels, output_channels, first_layer_stride, add_spatial=False):
        layers = []
        # First layer
        layers.append(Conv2dTranspose(input_channels, output_channels, kernel_size=3, stride=first_layer_stride, padding=1, output_padding=1))
        # Subsequent layers
        for _ in range(num_of_layers - 1):
            layers.append(Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1, residual=True))
        # Optional SpatialAttention
        if add_spatial:
            layers.append(SpatialAttention())  # Assumes SpatialAttention is a PyTorch Module
        
        return nn.Sequential(*layers)

    def diffuse(self, x, t, channels_to_mask):
        b, c, h, w = x.shape
        
        # Split spatial dimensions (bottom half only)
        split_idx = h // 2
        top_half = x[:, :, :split_idx, :]  # Entire top (all channels)
        bottom_half = x[:, :, split_idx:, :]  # Bottom to process
        
        # Split channels
        rgb_channels = bottom_half[:, :channels_to_mask, :, :]  # First 3 channels (R,G,B)
        other_channels = bottom_half[:, channels_to_mask:, :, :]  # Other channels (unchanged)

        # t的形状应为 [B]
        sqrt_alpha_t = torch.sqrt(self.alphas_cumprod[t])          # 自动广播为 [B,1,1,1]
        sqrt_one_minus_alpha_t = torch.sqrt(1 - self.alphas_cumprod[t])
        
        # 每个样本独立添加噪声
        epsilon = torch.randn_like(rgb_channels)
        noisy_rgb = sqrt_alpha_t.view(-1,1,1,1) * rgb_channels + sqrt_one_minus_alpha_t.view(-1,1,1,1) * epsilon

        # Recombine
        noisy_bottom = torch.cat([noisy_rgb, other_channels], dim=1)
        result = torch.cat([top_half, noisy_bottom], dim=2)


        return result
    
    def sample_t(self, expanded_B, probabilities):
        # Create a biased distribution towards higher values
        probabilities = torch.tensor([1, 1], dtype=torch.float32)
        probabilities = probabilities / probabilities.sum()
        t = torch.multinomial(probabilities, expanded_B, replacement=True)
        return t
    
    def forward(self, audio_sequences, face_sequences, use_face_enhancer=False, add_noise=True, noise_level=-1):
        
        B = audio_sequences.size(0)       
        input_dim_size = len(face_sequences.size())
        
        if input_dim_size > 4:
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)

        expanded_B = face_sequences.size(0)  # 展平后的 batch size

       
        t0 = self.sample_t(expanded_B, torch.tensor([0, 0, 0, 1], dtype=torch.float32)).to(face_sequences.device)
        t1 = self.sample_t(expanded_B, torch.tensor([0, 0, 1, 0], dtype=torch.float32)).to(face_sequences.device)
        t2 = self.sample_t(expanded_B, torch.tensor([0, 1, 0, 0], dtype=torch.float32)).to(face_sequences.device)
        t3 = self.sample_t(expanded_B, torch.tensor([1, 0, 0, 0], dtype=torch.float32)).to(face_sequences.device)
       
        self.alphas_cumprod = self.alphas_cumprod.to(face_sequences.device)
        
        
        # Obtain audio features
        audio_embedding1 = self.audio_encoder1(audio_sequences)
                  
        audio_embedding1 = self.audio_pos_encoder(audio_embedding1)
        
        height = face_sequences.size(2)
        width = face_sequences.size(3)
        top_face_mask = torch.ones_like(face_sequences)
        top_face_mask[:, :, :height // 2, :] = 0  # 将上半部分遮罩为0（黑色）
        top_face_sequences = face_sequences * top_face_mask  # 应用遮罩
          
        # if add_noise:
        #     face_sequences = self.diffuse(face_sequences.float(), t0, 3)
        
        # ----The face encoder-----
        face1 = self.face_encoder1(face_sequences)
        fed1 = self.fe_down1(face1)
        
        face1_bottom = self.face_encoder1_bottom(top_face_sequences)
        fed1_bottom = self.fe_down1_bottom(face1_bottom)
        
        fed1 = torch.cat([fed1, fed1_bottom], dim=1)
        
        # if add_noise:
        #   fed1 = self.diffuse(fed1.float(), t1, 16)

        face2 = self.face_encoder2(fed1)
        fed2 = self.fe_down2(face2)

        face2_moe1 = self.face_encoder2(fed1)
        fed2_moe1 = self.fe_down2_moe1(face2_moe1)

        fed2_combined = torch.cat([fed2, fed2_moe1], dim=1)

        # if add_noise:
        #   fed2 = self.diffuse(fed2.float(), t2, 128)

        face3 = self.face_encoder3(fed2_combined)
        fed3 = self.fe_down3(face3)

        
        # if add_noise:
        #   fed3 = self.diffuse(fed3.float(), t3, 256)

        face4 = self.face_encoder4(fed3)
        fed4 = self.fe_down4(face4)       
        
        face5 = self.face_encoder5(fed4)
        fed5 = self.fe_down5(face5)       
         
        fed5 = self.face_pos_encoder(fed5)
        
        FB, _, H, W = fed5.shape
        
        face_flatten = fed5.view(FB, 512, -1).permute(0, 2, 1)
        
        face_flatten = self.face_transformer_encoder(face_flatten)

        audio_flatten = audio_embedding1.view(FB, 512, -1).permute(0, 2, 1)
        
        audio_flatten = self.audio_transformer_encoder(audio_flatten)
        
        combined = torch.cat([face_flatten, audio_flatten], dim=1)
        
        
        bottlenet = self.bottlenet(combined)
        

        deface5 = self.face_decoder5(bottlenet)

        cat5 = torch.cat([deface5, face5], dim=1)
        cat5 = self.fd_conv5(cat5)
        
        cat5_with_skip = torch.cat([cat5, deface5], dim=1)
        deface4 = self.face_decoder4(cat5_with_skip)
        
        cat4 = torch.cat([deface4, face4], dim=1)
        cat4 = self.fd_conv4(cat4)

        cat4_with_skip = torch.cat([cat4, deface4], dim=1)
        deface3 = self.face_decoder3(cat4_with_skip)
        
        cat3 = torch.cat([deface3, face3], dim=1)
        cat3 = self.fd_conv3(cat3)

        cat3_with_skip = torch.cat([cat3, deface3], dim=1)
        deface2 = self.face_decoder2(cat3_with_skip)
        
        cat2 = torch.cat([deface2, face2 + face2_moe1], dim=1)
        cat2 = self.fd_conv2(cat2)
        
        cat2_with_skip = torch.cat([cat2, deface2], dim=1)
        deface1 = self.face_decoder1(cat2_with_skip)
        
        cat1 = torch.cat([deface1, face1 + face1_bottom], dim=1)
        cat1 = self.fd_conv1(cat1)

        x = self.output_block(cat1)
        
        outputs = x
        
        if input_dim_size > 4:
            x = torch.split(x, B, dim=0)
            outputs = torch.stack(x, dim=2)
        else:
            outputs = x
            
        return outputs, deface5, audio_embedding1

