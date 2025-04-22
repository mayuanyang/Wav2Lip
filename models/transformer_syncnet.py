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

class LearnablePositionalEncoding2D(nn.Module):
    def __init__(self, d_model: int, max_h: int, max_w: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 初始化一个可学习的位置编码参数，形状为 (1, d_model, max_h, max_w)
        self.pos_embedding = nn.Parameter(torch.randn(1, d_model, max_h, max_w))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征图，形状为 (B, C, H, W)
        Returns:
            加入可学习位置编码后的特征图，形状与输入相同
        """
        # 确保只取对应输入大小的部分
        _, _, H, W = x.shape
        return self.dropout(x + self.pos_embedding[:, :, :H, :W])
      
class TransformerSyncnet(nn.Module):
    def __init__(self, num_heads=8, num_encoder_layers=4):
        super(TransformerSyncnet, self).__init__()
        
        # --- Face encoder for individual frames ---
        self.face_encoder1 = nn.Sequential(
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
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            
            Conv2d(256, 256, kernel_size=3, stride=2, padding=1),  # Downsample
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            
            Conv2d(256, 256, kernel_size=3, stride=2, padding=1),  # Downsample
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            
            Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # Downsample
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            
            Conv2d(512, 256, kernel_size=3, stride=2, padding=1),  # Downsample
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
        )    
                
        # --- Audio encoder ---
        self.audio_encoder1 = nn.Sequential(
            # Example input shape: (B, 1, H_audio, W_audio)
            Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
        )

        self.audio_encoder2 = nn.Sequential(
            Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
        )
        
        self.audio_encoder3 = nn.Sequential(
            Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
        )
        
        self.audio_encoder4 = nn.Sequential(
            Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
        )
        
        self.face_pos_encoder = LearnablePositionalEncoding2D(d_model=256, max_h=1, max_w=2, dropout=0.1)
        self.audio_pos_encoder = LearnablePositionalEncoding2D(d_model=256, max_h=5, max_w=1, dropout=0.1)
        
        # 新增：各自模态的 self-attention 层
        
        self.face_self_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=num_heads, dropout=0.1, activation='gelu'),
            num_layers=4
        )
        
        self.audio_self_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=num_heads, dropout=0.1, activation='gelu'),
            num_layers=4
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=768, nhead=num_heads, dropout=0.1, activation='gelu'),
            num_layers=num_encoder_layers
        )
                
        
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
        num_of_frames = 5
        
        save_every_s_steps = 1000
        
        batch_size = face_embedding.shape[0]
        
        # --- Process audio modality ---
        audio_features1 = self.audio_encoder1(audio_embedding)  # (B, 512, H_a, W_a)        
        
        audio_features2 = self.audio_encoder2(audio_features1)
        
        audio_features3 = self.audio_encoder3(audio_features2)
        
        audio_features4 = self.audio_encoder4(audio_features3)
        
        audio_features4 = self.audio_pos_encoder(audio_features4)
        
        a_seq=audio_features4.view(batch_size, num_of_frames ,-1).permute(1, 0, 2) 
                        
        ### ---视觉分支--- ###
        face_embedding = face_embedding.view(batch_size * num_of_frames ,3 ,192 ,384)
        
        face1 = self.face_encoder1(face_embedding)
        face2 = self.face_encoder2(face1)
        face3 = self.face_encoder3(face2)
        face4 = self.face_encoder4(face3)
        face4 = self.face_pos_encoder(face4)
        #print('The face4', face4.shape)
        
        face_features = face4.flatten(1) #[b*5 ，512]
        face_seq = face_features.view(batch_size, 5, -1).permute(1, 0, 2)
        
        if step % save_every_s_steps == 0:
          self.save_sample_images(face1, 'face1', step)
          self.save_sample_images(face2, 'face2', step)
          self.save_sample_images(face3, 'face3', step)
          self.save_sample_images(face4, 'face4', step)
        
        
        face_self_out = self.face_self_attn(
            face_seq,
        )
        
        # Audio 的 self-attention
        audio_self_out = self.audio_self_attn(
            a_seq,
        )

        # --- 合并特征 ---
        combined = torch.cat((face_self_out, audio_self_out), dim=2)  # 沿特征维度拼接       
        
        attn_output = self.transformer_encoder(combined)
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