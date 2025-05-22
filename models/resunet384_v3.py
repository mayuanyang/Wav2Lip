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
    result = betas.clamp(min=0.001, max=0.85)

    #result = torch.cat([result, torch.zeros(1)])
    return result

    
class ResUNet384V3(nn.Module):
    def __init__(self):
        super(ResUNet384V3, self).__init__()
        
        num_diffusion_steps = 10
        self.num_diffusion_steps = num_diffusion_steps
        self.betas = cosine_noise_schedule(num_diffusion_steps)  # or cosine_noise_schedule()
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        self.face_encoder1 = nn.Sequential( #384x384
            Conv2d(12, 64, kernel_size=3, stride=1, padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
        )
        self.fe_down1 = nn.Sequential( 
            Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
        )#192x192
        
        self.face_encoder2 = nn.Sequential( 
            Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
        )
        self.fe_down2 = nn.Sequential(
            Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
        )#96x96

        self.face_encoder3 = nn.Sequential( 
            Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
        )
        self.fe_down3 = nn.Sequential( 
            Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
        )#48x48

        self.face_encoder4 = nn.Sequential( 
            Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
        )
        self.fe_down4 = nn.Sequential( 
            Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
        ) #24x24
        

        # --- Audio encoder ---
        self.audio_encoder1 = nn.Sequential(
            # Example input shape: (B, 1, H_audio, W_audio)
            Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
       
            Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
       
            Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
       
            Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
        )

        self.bottlenet = nn.Sequential(
            Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
        )
        
        self.face_transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8, dropout=0.1, activation='gelu'),
            num_layers=6
        )

        self.audio_transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8, dropout=0.1, activation='gelu'),
            num_layers=2
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8, dropout=0.1, activation='gelu'),
            num_layers=2
        )
        
        self.combined_layer = nn.Linear(1024, 512)
        
        # Decoders
        
        self.face_decoder4 = nn.Sequential( #48x48
            Conv2dTranspose(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
        )
        self.fd_conv4 = nn.Sequential(
            Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
        )

        self.face_decoder3 = nn.Sequential( #96x96
            Conv2dTranspose(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
        )
        self.fd_conv3 = nn.Sequential(
            Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
        )
        

        self.face_decoder2 = nn.Sequential( #192x192
            Conv2dTranspose(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
        )
        self.fd_conv2 = nn.Sequential(
            Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
        )


        self.face_decoder1 = nn.Sequential( #384x384
            Conv2dTranspose(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
        )
        
        self.fd_conv1 = nn.Sequential(
            Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
        )

        self.output_block = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        
        self.face_pos_encoder = LearnablePositionalEncoding2D(d_model=512, max_h=24, max_w=24, dropout=0.1)
        self.audio_pos_encoder = LearnablePositionalEncoding2D(d_model=512, max_h=24, max_w=24, dropout=0.1)
        


    def diffuse(self, x, t, noise_factor=0.5):
        b, c, h, w = x.shape
        
        # Split spatial dimensions (bottom half only)
        split_idx = h // 2
        top_half = x[:, :, :split_idx, :]  # Entire top (all channels)
        bottom_half = x[:, :, split_idx:, :]  # Bottom to process
        
        # Split channels
        rgb_channels = bottom_half[:, :3, :, :]  # First 3 channels (R,G,B)
        other_channels = bottom_half[:, 3:, :, :]  # Other channels (unchanged)

        # t的形状应为 [B]
        sqrt_alpha_t = torch.sqrt(self.alphas_cumprod[t])          # 自动广播为 [B,1,1,1]
        sqrt_one_minus_alpha_t = torch.sqrt(1 - self.alphas_cumprod[t])
        
        # 每个样本独立添加噪声
        epsilon = torch.randn_like(rgb_channels) * noise_factor
        noisy_rgb = sqrt_alpha_t.view(-1,1,1,1) * rgb_channels + sqrt_one_minus_alpha_t.view(-1,1,1,1) * epsilon

        # Recombine
        noisy_bottom = torch.cat([noisy_rgb, other_channels], dim=1)
        result = torch.cat([top_half, noisy_bottom], dim=2)


        return result
    
    def sample_t(self, expanded_B):
        # Create a biased distribution towards higher values
        probabilities = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.float32)
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

        if noise_level < 0:
          t = self.sample_t(expanded_B).to(face_sequences.device)
        else:
          t = [noise_level]
          print('Using noise', noise_level)

        self.alphas_cumprod = self.alphas_cumprod.to(face_sequences.device)
        
        
        # Obtain audio features
        audio_embedding1 = self.audio_encoder1(audio_sequences)
          
        audio_embedding1 = F.interpolate(audio_embedding1.float(), size=(24, 24), mode="bilinear")
        
        audio_embedding1 = self.audio_pos_encoder(audio_embedding1)
          
        if add_noise:
            face_sequences = self.diffuse(face_sequences.float(), t)
        
        # ----The face encoder-----
        face1 = self.face_encoder1(face_sequences)
        fed1 = self.fe_down1(face1)

        face2 = self.face_encoder2(fed1)
        fed2 = self.fe_down2(face2)

        face3 = self.face_encoder3(fed2)
        fed3 = self.fe_down3(face3)

        face4 = self.face_encoder4(fed3)
        fed4 = self.fe_down4(face4)
        fed4 = self.face_pos_encoder(fed4)
        
        
        FB, _, H, W = fed4.shape
        
        face_flatten = fed4.view(FB, 512, -1).permute(0, 2, 1)
        
        face_flatten = self.face_transformer_encoder(face_flatten)
        
        audio_flatten = audio_embedding1.view(FB, 512, -1).permute(0, 2, 1)

        audio_flatten = self.audio_transformer_encoder(audio_flatten)
        
        # Concatenate face and audio features
        combined = torch.cat((face_flatten, audio_flatten), dim=-1)
        
        # Apply linear transformation
        combined = self.combined_layer(combined)
        
        transformer_output = self.transformer_encoder(combined)
        swapped_back = transformer_output.permute(0, 2, 1)  # Shape: [5, 512, 9]
        original_shape = swapped_back.reshape(FB ,512 ,24 ,24 ) 
        
        
        # Get face bottleneck features
        bottleneck_input = torch.cat([fed4, original_shape], dim=1)
        bottlenet = self.bottlenet(bottleneck_input)

        deface4 = self.face_decoder4(bottlenet)
        cat4 = torch.cat([deface4, face4], dim=1)
        cat4 = self.fd_conv4(cat4)

        deface3 = self.face_decoder3(cat4)
        cat3 = torch.cat([deface3, face3], dim=1)
        cat3 = self.fd_conv3(cat3)
        
        deface2 = self.face_decoder2(cat3)
        cat2 = torch.cat([deface2, face2], dim=1)
        cat2 = self.fd_conv2(cat2)
        
        deface1 = self.face_decoder1(cat2)
        cat1 = torch.cat([deface1, face1], dim=1)
        cat1 = self.fd_conv1(cat1)

        x = self.output_block(cat1)
        

        outputs = x
        
        if input_dim_size > 4:
            x = torch.split(x, B, dim=0)
            outputs = torch.stack(x, dim=2)
        else:
            outputs = x
            
        return outputs, fed4, audio_embedding1

