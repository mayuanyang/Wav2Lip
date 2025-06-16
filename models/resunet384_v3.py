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
from torchvision.transforms import GaussianBlur
    
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
    return torch.tensor([0.6, 0.7, 0.8, 0.9])
    
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

        self.ellipse_params = {
            'center': (0.0, 0),  # (x,y)中心偏移(归一化坐标)
            'axes': (1, 0.75),      # (宽,高)比例
            'blur': 0.03             # 边缘模糊系数(相对于短边)
        }
              
        self.face_encoder1_full = self.construct_encoder_layers(4, 3, 64, 1, kernel=3)
        self.fe_down1_full = self.construct_encoder_layers(4, 64, 64, 2)
        
        self.face_encoder1_bottom = self.construct_encoder_layers(5, 9, 64, 1, kernel=3)
        self.fe_down1_bottom = self.construct_encoder_layers(4, 64, 64, 2)
                
        self.face_encoder2_full = self.construct_encoder_layers(4, 128, 256, 1)
        self.fe_down2_full = self.construct_encoder_layers(4, 256, 256, 2)
        
        self.face_encoder2_bottom = self.construct_encoder_layers(4, 64, 128, 1)
        self.fe_down2_bottom = self.construct_encoder_layers(4, 128, 128, 2)

        self.face_encoder3 = self.construct_encoder_layers(3, 640, 384, 1)
        self.fe_down3 = self.construct_encoder_layers(3, 384, 384, 2)

        self.face_encoder4 = self.construct_encoder_layers(3, 384, 384, 1)
        self.fe_down4 = self.construct_encoder_layers(3, 384, 384, 2)
        
        self.face_encoder5 = self.construct_encoder_layers(3, 384, 384, 1)
        self.fe_down5 = self.construct_encoder_layers(3, 384, 384, 2)

        # --- Audio encoder ---
        self.audio_encoder1 = nn.Sequential(
            # Example input shape: (B, 1, H_audio, W_audio)
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
       
            Conv2d(32, 64, kernel_size=3, stride=(2, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=(2,1), padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            
            Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
        )
        
        self.audio_adapter1 = nn.AdaptiveAvgPool2d((96, 96))
        #self.audio_adapter2 = nn.AdaptiveAvgPool2d((12, 12))
        

        self.bottlenet = self.construct_encoder_layers(2, 384, 384, 1, True)
                
        
        # Decoders
        self.face_decoder5 = self.construct_decoder_layers(3, 384, 384, 2)
        self.fd_conv5 = self.construct_encoder_layers(3, 768, 384, 1)
        
        self.face_decoder4 = self.construct_decoder_layers(3, 384, 384, 2)
        self.fd_conv4 = self.construct_encoder_layers(3, 768, 384, 1)
        
        self.face_decoder3 = self.construct_decoder_layers(3, 384, 384, 2)
        self.fd_conv3 = self.construct_encoder_layers(3, 768, 384, 1)
        
        self.face_decoder2 = self.construct_decoder_layers(4, 384, 384, 2)
        self.fd_conv2 = self.construct_encoder_layers(4, 640, 320, 1)

        self.face_decoder1 = self.construct_decoder_layers(4, 320, 128, 2)
        
        self.face_decoder0 = self.construct_encoder_layers(4, 192, 192, 1)
        
        self.fd_conv1 = self.construct_encoder_layers(4, 256, 64, 1)
        

        self.output_block = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        
        self.face_pos_encoder = LearnablePositionalEncoding2D(d_model=128, max_h=12, max_w=12, dropout=0.1)
        self.audio_pos_encoder = LearnablePositionalEncoding2D(d_model=128, max_h=12, max_w=12, dropout=0.1)
        
    def construct_encoder_layers(self, num_of_layers, input_channels, output_channels, first_layer_stride, add_spatial=False, kernel=3, add_pos_encoding=False, pos_h=0, pos_w=0, pos_d_model=64):
        layers = []
        padding = 1
        if kernel == 7:
          padding = 3
        # First layer
        layers.append(Conv2d(input_channels, output_channels, kernel_size=kernel, stride=first_layer_stride, padding=padding))
        # Subsequent layers
        for _ in range(num_of_layers - 1):
            layers.append(Conv2d(output_channels, output_channels, kernel_size=kernel, stride=1, padding=padding, residual=True))
            
        if add_pos_encoding:
            layers.append(LearnablePositionalEncoding2D(pos_d_model, pos_h, pos_w, 0.1))
        # Optional SpatialAttention
        if add_spatial:
            layers.append(SpatialAttention())  # Assumes SpatialAttention is a PyTorch Module
        
        return nn.Sequential(*layers)

    def construct_decoder_layers(self, num_of_layers, input_channels, output_channels, first_layer_stride, add_spatial=False, kernel=3, add_pos_encoding=False, pos_h=0, pos_w=0, pos_d_model=64):
        layers = []
        padding = 1
        if kernel == 7:
          padding = 3
        # First layer
        layers.append(Conv2dTranspose(input_channels, output_channels, kernel_size=kernel, stride=first_layer_stride, padding=padding, output_padding=1))
        # Subsequent layers
        for _ in range(num_of_layers - 1):
            layers.append(Conv2d(output_channels, output_channels, kernel_size=kernel, stride=1, padding=padding, residual=True))
        
        if add_pos_encoding:
            layers.append(LearnablePositionalEncoding2D(pos_d_model, pos_h, pos_w, 0.1))
        
        # Optional SpatialAttention
        if add_spatial:
            layers.append(SpatialAttention())  # Assumes SpatialAttention is a PyTorch Module
        
        return nn.Sequential(*layers)

    def generate_ellipse_mask(self, h, w, split_idx, device):
        """生成下半部分的椭圆遮罩"""
        # 创建下半部分归一化坐标网格 [-1,1]
        y_bottom = torch.linspace(0, 1, h - split_idx, device=device) * 2 - 1
        x_coord = torch.linspace(-1, 1, w, device=device)
        y_grid, x_grid = torch.meshgrid(y_bottom, x_coord, indexing='ij')
        
        # 应用椭圆方程
        dx, dy = self.ellipse_params['center']
        a_ratio, b_ratio = self.ellipse_params['axes']
        
        # 根据图像宽高比调整x轴比例
        adjusted_a = a_ratio * (w / h)  
        
        ellipse_mask = ((x_grid - dx)/adjusted_a)**2 + \
                      ((y_grid - dy)/b_ratio)**2 <= 1.0
        
        # 边缘模糊处理
        mask = ellipse_mask.float()
        if self.ellipse_params['blur'] > 0:
            kernel_size = int(min(h, w) * self.ellipse_params['blur']) | 1
            gaussian_blur = GaussianBlur(kernel_size=(kernel_size, kernel_size), sigma=kernel_size/3)
            mask = gaussian_blur(mask.unsqueeze(0).unsqueeze(0)).squeeze()
        
        return mask

    def diffuse(self, x, t, channels_to_mask=3):
        """
        带椭圆遮罩的扩散过程
        
        参数:
            x: [B,C,H,W]输入张量
            t: [B]时间步长 
            channels_to_mask: 需要处理的通道数
            
        返回:
            扩散后的张量，仅在下半部分椭圆区域内添加噪声
        """
        b, c, h, w = x.shape
        
        # Split空间维度(仅处理下半部分)
        split_idx = h // 2
        top_half = x[:, :, :split_idx, :]  # 上半部分(所有通道)
        bottom_half = x[:, :, split_idx:, :]  # 下半部分待处理
        
        # Split channels
        rgb_channels = bottom_half[:, :channels_to_mask, :, :]  # 前3通道(R,G,B)
        other_channels = bottom_half[:, channels_to_mask:, :, :]  # 其他通道(不变)
        
        # 生成椭圆遮罩 [H,W]
        mask = self.generate_ellipse_mask(h, w, split_idx, x.device)
        mask = mask.unsqueeze(0).unsqueeze(0).repeat(b, channels_to_mask, 1, 1)  # [B, channels_to_mask, H, W]
        
        # DDPM噪声扩散系数 
        sqrt_alpha_t = torch.sqrt(self.alphas_cumprod[t]).view(b, 1, 1, 1)  # 自动广播为 [B,1,1,1]
        sqrt_one_minus_alpha_t = torch.sqrt(1 - self.alphas_cumprod[t]).view(b, 1, 1, 1)
        
        # 每个样本独立添加噪声
        epsilon = torch.randn_like(rgb_channels)
        noisy_rgb = sqrt_alpha_t * rgb_channels + sqrt_one_minus_alpha_t * epsilon
        
        # Mask-aware噪声混合 (关键修改点!)
        noisy_rgb = rgb_channels * (1 - mask) + noisy_rgb * mask
        
        # Recombine各部分
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

        t0 = self.sample_t(expanded_B, torch.tensor([1, 1, 1, 1], dtype=torch.float32)).to(face_sequences.device)
       
        self.alphas_cumprod = self.alphas_cumprod.to(face_sequences.device)

        # Obtain audio features
        audio_embedding1 = self.audio_encoder1(audio_sequences)
        
        audio_adpter1_emb = self.audio_adapter1(audio_embedding1)
        
        #audio_adpter2_emb = self.audio_adapter2(audio_embedding1)
        
        first_3_channels = face_sequences[:, :3, :, :] 
        
        reference_bottom_half = face_sequences[:, 3:, 192:, :] 
        
        face_sequences1 = self.diffuse(first_3_channels.float(), t0, 3)
        
        # ----The face encoder-----
        face1_full = self.face_encoder1_full(face_sequences1)
        fed1_full = self.fe_down1_full(face1_full)
        
        padding = (0, 0, 192, 0)  # (left, right, top, bottom)
        
        face1_ref_bottom = self.face_encoder1_bottom(reference_bottom_half)
        face1_ref_bottom_padded = F.pad(face1_ref_bottom, padding, "constant", 0)
        
        fed1_ref_bottom = self.fe_down1_bottom(face1_ref_bottom_padded)

        fed1_concatenated = torch.cat([fed1_full, fed1_ref_bottom], dim=1)

        face2_full = self.face_encoder2_full(fed1_concatenated)
                
        fed2_full = self.fe_down2_full(face2_full)
        
        face2_ref_bottom = self.face_encoder2_bottom(fed1_ref_bottom)
        
        fed2_ref_bottom = self.fe_down2_bottom(face2_ref_bottom)
        
        fed2_concatenated = torch.cat([fed2_full, fed2_ref_bottom, audio_adpter1_emb], dim=1)

        face3 = self.face_encoder3(fed2_concatenated)
        fed3 = self.fe_down3(face3)

        face4 = self.face_encoder4(fed3)
        fed4 = self.fe_down4(fed3 + face4)       
        
        face5 = self.face_encoder5(fed4)
        fed5 = self.fe_down5(fed4 + face5)       
                 
        #fed5 = self.face_pos_encoder(fed5)
        
        #combined = torch.cat([fed5, audio_adpter2_emb], dim=1)
        
        bottlenet = self.bottlenet(fed5)
        
        deface5 = self.face_decoder5(bottlenet)

        cat5 = torch.cat([deface5, face5], dim=1)
        cat5 = self.fd_conv5(cat5)
        
        #cat5_with_skip = torch.cat([cat5, deface5], dim=1)
        deface4 = self.face_decoder4(cat5)
        
        cat4 = torch.cat([deface4, face4], dim=1)
        cat4 = self.fd_conv4(cat4)

        #cat4_with_skip = torch.cat([cat4, deface4], dim=1)
        deface3 = self.face_decoder3(cat4)
        
        cat3 = torch.cat([deface3, face3], dim=1)
        cat3 = self.fd_conv3(cat3)

        #cat3_with_skip = torch.cat([cat3, deface3], dim=1)
        deface2 = self.face_decoder2(cat3)
        
        cat2 = torch.cat([deface2, face2_full], dim=1)
        cat2 = self.fd_conv2(cat2)
        
        
        deface1 = self.face_decoder1(cat2)
        
        cat1 = torch.cat([face1_ref_bottom_padded, deface1], dim=1)
        deface1 = self.face_decoder0(cat1)
        
        cat1 = torch.cat([deface1, face1_full], dim=1)
        cat1 = self.fd_conv1(cat1)
        
        if step % 5000 == 0:       
          self.save_sample_images(face_sequences1, f'face_sequences1_{step}')
        
          self.save_sample_images(face1_full, f'face1_full_{step}')
          self.save_sample_images(fed1_full, f'fed1_full_{step}')
          
          self.save_sample_images(face1_ref_bottom_padded, f'face1_bottom_{step}')
          self.save_sample_images(fed1_ref_bottom, f'fed1_bottom_{step}')
          
          
          self.save_sample_images(face2_full, f'face2_full_{step}')
          self.save_sample_images(fed2_full, f'fed2_full_{step}')
          self.save_sample_images(fed2_ref_bottom, f'face2_bottom_{step}')
          self.save_sample_images(fed2_ref_bottom, f'fed2_bottom_{step}')
          
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
            
        return outputs, deface5, None

