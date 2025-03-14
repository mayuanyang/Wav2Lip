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

class MouthAttention(nn.Module):
    def __init__(self, in_channels, ellipse_ratio=0.4):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.ellipse_ratio = ellipse_ratio  # 椭圆半轴占高度/宽度的比例（例如0.3对应60%区域）
        self.mask_weight = nn.Parameter(torch.tensor(1.0))  # 可学习的mask权重
        self.expansion_factor = nn.Parameter(torch.tensor(1.0))  # 可学习的扩展因子

    def forward(self, input):
        B, _, H, W = input.size()
        device = input.device

        # 生成网格坐标
        y, x = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )
        y = y.float()
        x = x.float()

        # 计算椭圆中心和半轴长度
        center_y = H / 2.0
        center_x = W / 2.0
        a = self.ellipse_ratio * H * self.expansion_factor  # y轴半长轴（高度方向），乘以扩展因子
        b = self.ellipse_ratio * W * self.expansion_factor  # x轴半短轴（宽度方向），乘以扩展因子

        # 计算椭圆方程条件
        mask_condition = ((y - center_y)**2 / (a**2)) + ((x - center_x)**2 / (b**2)) <= 1.0
        mask = mask_condition.float().unsqueeze(0).unsqueeze(0)  # 扩展为[B, 1, H, W]

        # 扩展到批量大小
        mask = mask.repeat(B, 1, 1, 1)

        # 使用可学习的权重控制mask的影响
        attn = self.sigmoid(self.conv(input))
        attn = attn * mask  # 保留椭圆区域的注意力
        # 添加可学习的权重，允许梯度流动
        weighted_mask = self.mask_weight * mask
        return input * (attn + weighted_mask)  # 或直接返回 x * (attn * weighted_mask)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=5, num_layers=2):
        super(SpatialAttention, self).__init__()
        layers = []
        padding = (kernel_size - 1) // 2
        # Start with 2-channel input (from avg and max pooling)
        in_channels = 2
        hidden_channels = 4  # Arbitrary choice; adjust as needed
        
        # Add intermediate layers
        for _ in range(num_layers - 1):
            layers.append(nn.Conv2d(in_channels, hidden_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.ReLU(inplace=True))
            in_channels = hidden_channels
        
        # Final convolution to get a single-channel attention map
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=kernel_size, padding=padding, bias=False))
        self.conv = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()
        # Learnable scaling factor to adjust the contribution of the attention
        self.alpha = nn.Parameter(torch.zeros(1))  # Initialized to zero (or a small value)
    
    
    def forward(self, x):
        # x: (B, C, H, W)
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)
        x_cat = torch.cat([avg_pool, max_pool], dim=1)     # (B, 2, H, W)
        attn = self.conv(x_cat)
        attn = self.sigmoid(attn)
        out = x + self.alpha * attn
        return out


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
            Conv2d(15, 64, kernel_size=3, stride=2, padding=1, leaking=0.1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, leaking=0.1, residual=True), 
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, leaking=0.1, residual=True), 
            #MouthAttention(128),
            #SpatialAttention()
        )
        
        self.face_encoder2 = nn.Sequential(
            Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Downsample
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            #MouthAttention(256),
            #SpatialAttention()
        )
        
        self.face_encoder3 = nn.Sequential(
            Conv2d(128, 128, kernel_size=3, stride=2, padding=1),  # Downsample width
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            #SpatialAttention()
        )
        
        self.face_encoder4 = nn.Sequential(
            Conv2d(128, 128, kernel_size=3, stride=2, padding=1),  # Downsample
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
        )
        
        self.face_skip1 = nn.Sequential(    
            Conv2d(128, 256, kernel_size=3, stride=2, padding=1, leaking=0.05),  # Downsample
        )
        
        self.face_skip2 = nn.Sequential(    
            Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # Downsample
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
        )
        
        self.face_skip3 = nn.Sequential(    
            Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
        )
        
        # --- Audio encoder ---
        self.audio_encoder1 = nn.Sequential(
            # Example input shape: (B, 1, H_audio, W_audio)
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1, leaking=0.05),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, leaking=0.05, residual=True),
            
        )

        self.audio_encoder2 = nn.Sequential(
            Conv2d(32, 64, kernel_size=3, stride=(2,1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
        )
        
        self.audio_encoder3 = nn.Sequential(
            Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
        )
        
        
        self.pos_encoder = PositionalEncoding2D(128, 20, 24)
        
        # Projection layers: ensure both modalities have the same embedding dimension.
        # Here, both face and audio encoders output 512 channels.
        self.face_proj = nn.Conv2d(128, embed_dim, kernel_size=1)
        self.audio_proj = nn.Conv2d(128, embed_dim, kernel_size=1)

        self.face_layer_norm = nn.LayerNorm(embed_dim)
        self.audio_layer_norm = nn.LayerNorm(embed_dim)
        
        self.cross_attn = CrossModalAttention2d(128)
        self.fuse = nn.Sequential(
            Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
        )
                
        
        # Cross-modal Transformer encoder.
        # We will first flatten the spatial dimensions into a token sequence.
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.cross_modal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        self.relu = nn.LeakyReLU(0.01, inplace=False)
        
        # Final classification head.
        # We pool tokens for each modality separately, then concatenate their global features.
        self.classifier = nn.Sequential(
            nn.Linear(1152, 64),
            nn.LeakyReLU(0.01, inplace=False),
            nn.Linear(64, 1)  # binary classification output
        )
        
    def pad_to_shape(self, tensor, target_shape):
        """
        Pads a tensor to target_shape with center alignment.
        
        Args:
            tensor (torch.Tensor): input tensor of shape (B, C, H, W)
            target_shape (tuple): target shape (target_H, target_W)
        
        Returns:
            padded_tensor (torch.Tensor): tensor padded to shape (B, C, target_H, target_W)
        """
        B, C, H, W = tensor.shape
        target_H, target_W = target_shape

        # Calculate padding for height (top, bottom)
        pad_top = (target_H - H) // 2 if target_H > H else 0
        pad_bottom = target_H - H - pad_top if target_H > H else 0

        # Calculate padding for width (left, right)
        pad_left = (target_W - W) // 2 if target_W > W else 0
        pad_right = target_W - W - pad_left if target_W > W else 0

        # F.pad expects padding as (pad_left, pad_right, pad_top, pad_bottom)
        padded_tensor = F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        return padded_tensor
      
    def forward(self, face_embedding, audio_embedding, step):
        """
        face_embedding: tensor of shape (B, 15, H, W) -> 5 images concatenated (each 3 channels)
        audio_embedding: tensor of shape (B, 1, H_audio, W_audio)
        """
        
        #audio_embedding = F.interpolate(audio_embedding, size=(320, 64), mode='bicubic', align_corners=False)
        # --- Process face modality ---
        
        face1 = self.face_encoder1(face_embedding)
        if step % 5000 == 0:
          self.save_sample_images(face1, 'face1', step)
        
        # face_skip1 = self.face_skip1(face1)
        
        # face_skip2 = self.face_skip2(face_skip1)
        
        # face_skip3 = self.face_skip3(face_skip2)
        
        face2 = self.face_encoder2(face1)
        if step % 5000 == 0:
          self.save_sample_images(face2, 'face2', step)
        
        #face2 = face2 + face_skip1
                        
        face3 = self.face_encoder3(face2)
        if step % 5000 == 0:
          self.save_sample_images(face3, 'face3', step)

        #face3 = face3 + face_skip2
        
        face4 = self.face_encoder4(face3)
        
        if step % 5000 == 0:
          self.save_sample_images(face4, 'face4', step)
        
        #face_features = face4 + face_skip3  # (B, 512, H_f, W_f)
        face_features = face4  # (B, 512, H_f, W_f)
        
        # --- Process audio modality ---
        audio_features1 = self.audio_encoder1(audio_embedding)  # (B, 512, H_a, W_a)
        if step % 5000 == 0:
          self.save_sample_images(audio_features1, 'audio1', step)
        
        
        audio_features2 = self.audio_encoder2(audio_features1)
        
        audio_features3 = self.audio_encoder3(audio_features2)

        audio_features = audio_features3

               

        # torch.Size([2, 512, 12, 24]) torch.Size([2, 512, 20, 8])
        target_shape = (20, 24)  # (20, 24) in this case
        face_features = self.pad_to_shape(face_features, target_shape)
        audio_features = self.pad_to_shape(audio_features, target_shape)
        
        
        
        #print('The face and audio shape', face_features.shape, audio_features.shape)
        
        # --- Apply positional encoding separately to each modality ---
        
        face_features = self.pos_encoder(face_features)
        audio_features = self.pos_encoder(audio_features)
        
                
        B, _, H, W = face_features.shape
        
        out = self.cross_attn(face_features, audio_features)
        
        fused = self.fuse(out)
        
        out_proj = fused.view(B, -1)
        #out_proj = out_proj.permute(B, -1)
        #print('The out shape 2', out_proj.shape)
        
        result = self.classifier(out_proj)
        
        
        if step % 1000 == 0:
          self.save_sample_images(face_features, 'face_final', step)
          self.save_sample_images(audio_features, 'audio_final', step)
        
        
        
        
                
        # # --- Project both modalities to the common embedding dimension ---
        # face_proj = self.face_proj(face_features)   # (B, embed_dim, H, W)
        # audio_proj = self.audio_proj(audio_features)  # (B, embed_dim, H, W)
        
        # # --- Flatten spatial dimensions into tokens ---
        # # Resulting shape: (num_tokens, B, embed_dim) where num_tokens = H * W.
        # face_tokens = face_proj.view(B, -1, H * W).permute(2, 0, 1)
        # audio_tokens = audio_proj.view(B, -1, H * W).permute(2, 0, 1)

        # face_tokens = self.face_layer_norm(face_tokens)
        # audio_tokens = self.audio_layer_norm(audio_tokens)
        
        
        # # --- Concatenate the token sequences along the token dimension ---
        # # Combined tokens shape: (2 * num_tokens, B, embed_dim)
        # combined_tokens = torch.cat([face_tokens, audio_tokens], dim=0)
        
        
        
        # # --- Apply cross-modal Transformer encoder ---
        # combined_tokens = self.cross_modal_transformer(combined_tokens)
        # combined_tokens = self.relu(combined_tokens)
        
        # # --- Separate tokens back by modality ---
        # num_tokens = H * W
        # face_tokens_out = combined_tokens[:num_tokens, :, :]  # (num_tokens, B, embed_dim)
        # audio_tokens_out = combined_tokens[num_tokens:, :, :]   # (num_tokens, B, embed_dim)
        
        # # --- Global average pooling for each modality ---
        # #face_global = face_tokens_out.max(dim=0)[0]   # (B, embed_dim)
        # #audio_global = audio_tokens_out.max(dim=0)[0] # (B, embed_dim)
        
        # face_global = torch.cat([face_tokens_out.mean(dim=0), face_tokens_out.max(dim=0)[0]], dim=1)
        # audio_global = torch.cat([audio_tokens_out.mean(dim=0), audio_tokens_out.max(dim=0)[0]], dim=1)

        
        # # --- Fuse modalities ---
        # fused_features = torch.cat([face_global, audio_global], dim=1)  # (B, 2*embed_dim)
        
        # # --- Classification head ---
        # logits = self.classifier(fused_features)  # (B, 1)
        
        return result, None, None

    def save_sample_images(self, x, layer, step):
        
        B, C, H, W = x.shape
        for b in range(B):
          if b > 10:
            break


          # Extract the b-th sample: shape (256, H, W)
          sample = x[b]
          # Convert to numpy and rearrange to shape (H, W, 256)
          sample_np = sample.detach().cpu().numpy().transpose(1, 2, 0)  # (H, W, 256)
          # Reshape to (H*W, 256) so that each pixel is a 256-d vector
          pixels = sample_np.reshape(-1, C)  # (H*W, 256)

          # Apply PCA to reduce 256 channels to 3
          pca = PCA(n_components=3)
          pixels_pca = pca.fit_transform(pixels)  # (H*W, 3)

          # Normalize the PCA output to range 0-255 for visualization
          min_vals = pixels_pca.min(axis=0)
          max_vals = pixels_pca.max(axis=0)
          pixels_norm = (pixels_pca - min_vals) / (max_vals - min_vals + 1e-8) * 255.0
          pixels_norm = pixels_norm.astype(np.uint8)

          # Reshape back to an image with shape (H, W, 3)
          image_rgb = pixels_norm.reshape(H, W, 3)
          
          # OpenCV expects images in BGR order. If needed, convert RGB to BGR.
          image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
          
          # Save the image
          filename = os.path.join("", f"{layer}_{step}_sample_{b}_combined.png")
          cv2.imwrite(filename, image_bgr)
        
        
        grid_rows, grid_cols = 16, 16  # 16x16 = 256 channels

        for b in range(B):
            if b > 3:
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
            grid_filename = os.path.join("", f"{layer}_sample_{b}_grid.png")
            cv2.imwrite(grid_filename, grid_img)
        