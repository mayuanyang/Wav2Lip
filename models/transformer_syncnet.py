import torch
from torch import nn, resize_as_
from torch.nn import functional as F
from .conv import Conv2d
from .self_attention import AttentionBlock
from .cross_modal_attention import CrossModalAttention2d
import torch.nn.init as init
import numpy as np
import cv2, os
from sklearn.decomposition import PCA

def check_nan(tensor, name):
    if torch.isnan(tensor).any():
        print('NaN problem', f"NaN in {name}")

def initialize_weights(module):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
            #print('Init')

class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model: int, max_h: int, max_w: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建二维位置编码张量 (H, W, d_model)
        pe = torch.zeros(max_h, max_w, d_model)
        
        # 行方向的位置编码（height）
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pos_row = torch.arange(max_h).unsqueeze(1)  # 形状：(H, 1)
        # 广播相乘：(H,1) * (d_model/2,) → (H, d_model/2)
        pe[:, :, 0::2] = torch.sin(pos_row * div_term).unsqueeze(1)  # 扩展为 (H, 1, d_model/2)
        
        # 列方向的位置编码（width）
        pos_col = torch.arange(max_w).unsqueeze(0)  # 形状：(1, W)
        # 广播相乘：(1, W) * (d_model/2,) → (W, d_model/2) → 需要调整维度
        pe[:, :, 1::2] = torch.cos(pos_col.unsqueeze(-1) * div_term.unsqueeze(0))  # 关键修改
        
        # 调整形状为 (1, d_model, H, W)
        self.register_buffer('pe', pe.permute(2, 0, 1).unsqueeze(0))  # 形状：(1, d_model, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征图，形状为 (B, C, H, W)
        Returns:
            编码后的位置特征，形状与输入相同
        """
        # 确保位置编码的 H 和 W 不超过输入的 H 和 W
        _, _, H, W = x.shape
        return self.dropout(x + self.pe[:, :, :H, :W])
      
class TransformerSyncnet(nn.Module):
    def __init__(self, num_heads=8, num_encoder_layers=4, embed_dim=512):
        super(TransformerSyncnet, self).__init__()
        
        self.embed_dim = embed_dim
        # --- Face encoder for individual frames ---
        self.face_encoder1 = nn.Sequential(
            # Input: (B, 15, H, W)  where 15 = 5 images x 3 channels
            Conv2d(3, 128, kernel_size=3, stride=2, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
        )
        
        self.face_encoder2 = nn.Sequential(
            Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # Downsample
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
        )
        
        self.face_encoder3 = nn.Sequential(
            Conv2d(256, 256, kernel_size=3, stride=2, padding=1),  # Downsample width
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
        )
        
        self.face_encoder4 = nn.Sequential(
            Conv2d(256, 256, kernel_size=3, stride=2, padding=1),  # Downsample
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            
            Conv2d(256, 256, kernel_size=3, stride=2, padding=1),  # Downsample
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            
            Conv2d(256, 256, kernel_size=3, stride=2, padding=1),  # Downsample
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            
            Conv2d(256, 256, kernel_size=3, stride=2, padding=1),  # Downsample
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            
            Conv2d(256, 256, kernel_size=3, stride=2, padding=1),  # Downsample
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
        )    
                
        # --- Audio encoder ---
        self.audio_encoder1 = nn.Sequential(
            # Example input shape: (B, 1, H_audio, W_audio)
            Conv2d(1, 64, kernel_size=3, stride=2, padding=1, leaking=0.05),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
        )

        self.audio_encoder2 = nn.Sequential(
            Conv2d(64, 128, kernel_size=3, stride=2, padding=1, leaking=0.05),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
        )
        
        self.audio_encoder3 = nn.Sequential(
            Conv2d(128, 256, kernel_size=3, stride=2, padding=1, leaking=0.05),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
        )
        
        self.audio_encoder4 = nn.Sequential(
            Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
        )        
        
        self.face_pos_encoder = PositionalEncoding2D(512, 12, 24)
        self.audio_pos_encoder = PositionalEncoding2D(512, 20, 4)
                
        
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(768, num_heads, dropout=0.1) 
            for _ in range(num_encoder_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(768) for _ in range(num_encoder_layers)
        ])
        
        self.fusion_proj = nn.Linear(768 ,256)
        
        
        # Final classification head.
        # We pool tokens for each modality separately, then concatenate their global features.
        self.classifier = nn.Sequential(
            # nn.Linear(1536, 512), for 192
            nn.Linear(3840, 256), 
            nn.LeakyReLU(0.2, inplace=False),
            nn.Dropout(p=0.1),
            nn.Linear(256, 1)  # binary classification output
        )
      
        self.face_encoder1.apply(initialize_weights)
        self.face_encoder2.apply(initialize_weights)
        self.face_encoder3.apply(initialize_weights)
        self.face_encoder4.apply(initialize_weights)        
        
        self.audio_encoder1.apply(initialize_weights)
        self.audio_encoder2.apply(initialize_weights)
        self.audio_encoder3.apply(initialize_weights)
        self.audio_encoder4.apply(initialize_weights)

        
        #self.reduce.apply(initialize_weights)
        self.classifier.apply(initialize_weights)
      
    def forward(self, face_embedding, audio_embedding, step):
        """
        face_embedding: tensor of shape (B, 15, H, W) -> 5 images concatenated (each 3 channels)
        audio_embedding: tensor of shape (B, 1, H_audio, W_audio)
        """
        
        save_every_s_steps = 2
        
        batch_size = face_embedding.shape[0]

        # Calculate min and max values
        min_value = torch.min(audio_embedding)
        max_value = torch.max(audio_embedding)

        epsilon = 1e-6

        # Normalize the tensor to range (0, 1)
        audio_embedding = (audio_embedding - min_value) / (max_value - min_value + epsilon)

        # Scale to ensure it does not reach exactly 0 or 1
        audio_embedding = audio_embedding * (1 - epsilon) + epsilon
        
        # --- Process audio modality ---
        audio_features1 = self.audio_encoder1(audio_embedding)  # (B, 512, H_a, W_a)        
        
        audio_features2 = self.audio_encoder2(audio_features1)
        
        audio_features3 = self.audio_encoder3(audio_features2)
        
        audio_features4 = self.audio_encoder4(audio_features3)
        
        
        a_seq=audio_features4.view(batch_size, 5 ,-1).permute(1, 0, 2) 
        a_seq_flat = a_seq.flatten(1)
        a_seq_flat = a_seq_flat.view(batch_size, 5, -1).permute(1, 0, 2)
                
        ### ---视觉分支--- ###
                
        face_embedding = face_embedding.view(batch_size *5 ,3 ,192 ,384)
        
        face1 = self.face_encoder1(face_embedding)
        face2 = self.face_encoder2(face1)
        face3 = self.face_encoder3(face2)
        face4 = self.face_encoder4(face3)
        face_features = face4.flatten(1) #[b*5 ，512]
        face_seq = face_features.view(batch_size, 5, -1).permute(1, 0, 2)
        
        if step % save_every_s_steps == 0:
          self.save_sample_images(face1, 'face1', step)
          self.save_sample_images(face2, 'face2', step)
          self.save_sample_images(face3, 'face3', step)
          self.save_sample_images(face4, 'face4', step)
        
        combined = torch.cat((face_seq, a_seq_flat), dim=2)
        
        attn_output = combined
        
        for layer, layer_norm in zip(self.attn_layers, self.layer_norms):
            # Apply multi-head attention
            attn_output, _ = layer(attn_output, combined, combined)
            # Apply LayerNorm after attention
            attn_output = layer_norm(attn_output + combined)
        
        attn_output = attn_output.permute(1, 0, 2).reshape(batch_size, -1)
        
        output = self.classifier(attn_output)
        
        return output, None, None

    def save_sample_images(self, x, layer, step):
        base_dir = 'temp'
        B, C, H, W = x.shape
                
        
        grid_rows, grid_cols = 16, 16  # 16x16 = 256 channels

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
            grid_filename = os.path.join(base_dir, f"{layer}_{step}_sample_{b}_grid.png")
            cv2.imwrite(grid_filename, grid_img)