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
import numpy as np
    
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
    return torch.tensor([0.1, 0.5, 0.8, 0.9])
    
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
        return x * (1 + x_att)
      
class ResUNet384V3(nn.Module):
    def __init__(self):
        super(ResUNet384V3, self).__init__()
        
        num_diffusion_steps = 2
        self.num_diffusion_steps = num_diffusion_steps
        self.betas = linear_schedule()  # or cosine_noise_schedule()
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
              
        self.face_encoder1_moe1 = self.construct_encoder_layers(3, 12, 64, 1)
        self.fe_down1_moe1 = self.construct_encoder_layers(3, 64, 64, 2)
        
        self.face_encoder1_moe2 = self.construct_encoder_layers(3, 12, 64, 1)
        self.fe_down1_moe2 = self.construct_encoder_layers(3, 64, 64, 2)
        
        self.face_encoder1_moe3 = self.construct_encoder_layers(3, 12, 64, 1)
        self.fe_down1_moe3 = self.construct_encoder_layers(3, 64, 64, 2)
        
        self.face_encoder1_moe4 = self.construct_encoder_layers(3, 12, 64, 1)
        self.fe_down1_moe4 = self.construct_encoder_layers(3, 64, 64, 2)

        self.face_encoder1_moe5 = self.construct_encoder_layers(3, 12, 64, 1)
        self.fe_down1_moe5 = self.construct_encoder_layers(3, 64, 64, 2)
        
        
        self.face_encoder2_moe1 = self.construct_encoder_layers(3, 320, 128, 1, True)
        self.fe_down2_moe1 = self.construct_encoder_layers(3, 128, 128, 2)
        
        self.face_encoder2_moe2 = self.construct_encoder_layers(3, 320, 128, 1, True)
        self.fe_down2_moe2 = self.construct_encoder_layers(3, 128, 128, 2)

        self.face_encoder3 = self.construct_encoder_layers(3, 256, 128, 1, True)
        self.fe_down3 = self.construct_encoder_layers(3, 128, 128, 2, True)

        self.face_encoder4 = self.construct_encoder_layers(3, 128, 128, 1, True)
        self.fe_down4 = self.construct_encoder_layers(3, 128, 128, 2)
        
        self.face_encoder5 = self.construct_encoder_layers(3, 128, 128, 1, True)
        self.fe_down5 = self.construct_encoder_layers(3, 128, 128, 2)
        
        # self.face_transformer_encoder = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(d_model=128, nhead=8, dropout=0.1, activation='gelu'),
        #     num_layers=2
        # )

        # --- Audio encoder ---
        self.audio_encoder1 = nn.Sequential(
            # Example input shape: (B, 1, H_audio, W_audio)
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
       
            Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
       
            Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
       
            nn.AdaptiveAvgPool2d((12, 12))
        )
        

        self.bottlenet = self.construct_encoder_layers(2, 256, 128, 1, True)
                
        
        # Decoders
        self.face_decoder5 = self.construct_decoder_layers(3, 128, 128, 2, True)
        self.fd_conv5 = self.construct_encoder_layers(3, 256, 128, 1, True)
        
        self.face_decoder4 = self.construct_decoder_layers(3, 256, 128, 2, True)
        self.fd_conv4 = self.construct_encoder_layers(3, 256, 128, 1, True)
        
        self.face_decoder3 = self.construct_decoder_layers(3, 256, 128, 2, True)
        self.fd_conv3 = self.construct_encoder_layers(3, 256, 128, 1, True)
        
        self.face_decoder2 = self.construct_decoder_layers(3, 256, 128, 2, True)
        self.fd_conv2 = self.construct_encoder_layers(3, 384, 128, 1, True)

        self.face_decoder1 = self.construct_decoder_layers(3, 256, 64, 2, True)
        self.fd_conv1 = self.construct_encoder_layers(3, 384, 64, 1, True)
        

        self.output_block = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        
        self.face_pos_encoder = LearnablePositionalEncoding2D(d_model=128, max_h=12, max_w=12, dropout=0.1)
        self.audio_pos_encoder = LearnablePositionalEncoding2D(d_model=128, max_h=12, max_w=12, dropout=0.1)
        
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
        probabilities = probabilities / probabilities.sum()
        t = torch.multinomial(probabilities, expanded_B, replacement=True)
        return t
    
    def save_sample_images(self, x, prefix):
        base_dir = 'temp'
        B, C, H, W = x.shape
                
        
        grid_rows, grid_cols = 4, 3  # 16x16 = 256 channels

        for b in range(B):
            if b > 2:
              break

            # Create an empty grid image (grayscale)
            grid_img = np.zeros((grid_rows * H, grid_cols * W), dtype=np.uint8)
            for c in range(C):
                if c < grid_cols * grid_rows:
                  # Convert the channel to a NumPy array
                  img = x[b, c].detach().cpu().numpy()  # (H, W)
                  img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8) * 255.0
                  img = img.astype(np.uint8)
                  # Compute grid position
                  row = c // grid_cols
                  col = c % grid_cols
                  grid_img[row*H:(row+1)*H, col*W:(col+1)*W] = img
            grid_filename = os.path.join(base_dir, f"{prefix}_sample_{b}_grid.png")
            cv2.imwrite(grid_filename, grid_img)
            
    def forward(self, audio_sequences, face_sequences, step):
        
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
        
        min_val = audio_sequences.min()
        max_val = audio_sequences.max()
        audio_sequences_scaled = (audio_sequences - min_val) / (max_val - min_val) * 2 - 1
        audio_sequences_normalized = (audio_sequences_scaled + 1) / 2

        # Obtain audio features
        audio_embedding1 = self.audio_encoder1(audio_sequences_normalized)
        audio_embedding1 = self.audio_pos_encoder(audio_embedding1)        
          
        face_sequences1 = self.diffuse(face_sequences.float(), t0, 3)
        face_sequences2 = self.diffuse(face_sequences.float(), t1, 3)
        face_sequences3 = self.diffuse(face_sequences.float(), t2, 3)
        face_sequences4 = self.diffuse(face_sequences.float(), t3, 3)
        face_sequences5 = face_sequences
        
        # ----The face encoder-----
        face1_moe1 = self.face_encoder1_moe1(face_sequences1)
        fed1_moe1 = self.fe_down1_moe1(face1_moe1)
        
        face1_moe2 = self.face_encoder1_moe2(face_sequences2)
        fed1_moe2 = self.fe_down1_moe2(face1_moe2)
        
        face1_moe3 = self.face_encoder1_moe3(face_sequences3)
        fed1_moe3 = self.fe_down1_moe3(face1_moe3)
        
        face1_moe4 = self.face_encoder1_moe4(face_sequences4)
        fed1_moe4 = self.fe_down1_moe4(face1_moe4)

        face1_moe5 = self.face_encoder1_moe5(face_sequences5)
        fed1_moe5 = self.fe_down1_moe5(face1_moe5)
        
        fed1_concatenated = torch.cat([fed1_moe1, fed1_moe2, fed1_moe3, fed1_moe4, fed1_moe5], dim=1)
        

        face2_moe1 = self.face_encoder2_moe1(fed1_concatenated)
        fed2_moe1 = self.fe_down2_moe1(face2_moe1)

        face2_moe2 = self.face_encoder2_moe2(fed1_concatenated)
        fed2_moe2 = self.fe_down2_moe2(face2_moe2)

        fed2_concatenated = torch.cat([fed2_moe1, fed2_moe2], dim=1)

        face3 = self.face_encoder3(fed2_concatenated)
        fed3 = self.fe_down3(face3)
        
        

        face4 = self.face_encoder4(fed3)
        fed4 = self.fe_down4(fed3 + face4)       
        
        face5 = self.face_encoder5(fed4)
        fed5 = self.fe_down5(fed4 + face5)       
        
                 
        fed5 = self.face_pos_encoder(fed5)
        
        # FB, _, H, W = fed5.shape
        
        # face_flatten = fed5.view(FB, 128, -1).permute(0, 2, 1)
        
        # face_output = self.face_transformer_encoder(face_flatten)

        # face_output = face_flatten + face_output
        
        # face_swapped_back = face_output.permute(0, 2, 1).view(FB, 128, 12, 12)  # Shape: [5, 512, 144]

        #self.save_sample_images(face_swapped_back, 10)
        combined = torch.cat([fed5, audio_embedding1], dim=1)
        
        
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
        
        cat2 = torch.cat([deface2, face2_moe1, face2_moe2], dim=1)
        cat2 = self.fd_conv2(cat2)
        
        cat2_with_skip = torch.cat([cat2, deface2], dim=1)
        deface1 = self.face_decoder1(cat2_with_skip)
        
        cat1 = torch.cat([deface1, face1_moe1, face1_moe2, face1_moe3, face1_moe4, face1_moe5], dim=1)
        cat1 = self.fd_conv1(cat1)
        
        if step % 5000 == 0:       
          self.save_sample_images(face_sequences1, f'face_sequences1_{step}')
          self.save_sample_images(face_sequences5, f'face_sequences5_{step}')
        
          self.save_sample_images(face1_moe1, f'face1_moe1_{step}')
          self.save_sample_images(fed1_moe1, f'fed1_moe1_{step}')
          
          self.save_sample_images(face1_moe2, f'face1_moe2_{step}')
          self.save_sample_images(fed1_moe2, f'fed1_moe2_{step}')
          
          self.save_sample_images(face1_moe3, f'face1_moe3_{step}')
          self.save_sample_images(fed1_moe3, f'fed1_moe3_{step}')
          
          self.save_sample_images(face1_moe4, f'face1_moe4_{step}')
          self.save_sample_images(fed1_moe4, f'fed1_moe4_{step}')

          self.save_sample_images(face1_moe5, f'face1_moe5_{step}')
          self.save_sample_images(fed1_moe5, f'fed1_moe5_{step}')
          
          self.save_sample_images(face2_moe1, f'face2_moe1_{step}')
          self.save_sample_images(fed2_moe1, f'fed2_moe1_{step}')
          self.save_sample_images(face2_moe2, f'face2_moe2_{step}')
          self.save_sample_images(fed2_moe2, f'fed2_moe2_{step}')
          
          self.save_sample_images(face3, f'face3_{step}')
          self.save_sample_images(fed3, f'fed3_{step}')
          
          #-----Decoder------
          self.save_sample_images(deface4, f'deface4_{step}')
          self.save_sample_images(deface3, f'deface3_{step}')
          self.save_sample_images(deface2, f'deface2_{step}')
          self.save_sample_images(deface1, f'deface1_{step}')
          

        x = self.output_block(cat1)
        
        outputs = x
        
        if input_dim_size > 4:
            x = torch.split(x, B, dim=0)
            outputs = torch.stack(x, dim=2)
        else:
            outputs = x
            
        return outputs, deface5, audio_embedding1

