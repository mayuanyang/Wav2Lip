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

class MouthAttention(nn.Module):
    def __init__(self, in_channels, ellipse_ratio=0.5):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.ellipse_ratio = ellipse_ratio  # 椭圆半轴占高度/宽度的比例（例如0.3对应60%区域）
        self.mask_weight = nn.Parameter(torch.tensor(1.0))  # 可学习的mask权重
        self.expansion_factor = nn.Parameter(torch.tensor(0.5))  # 可学习的扩展因子

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
    def __init__(self, kernel_size=5, num_layers=2, hidden_channels=8):
        super(SpatialAttention, self).__init__()
        layers = []
        padding = (kernel_size - 1) // 2
        # Start with 2-channel input (from avg and max pooling)
        in_channels = 2

        # Add intermediate layers
        for _ in range(num_layers - 1):
            layers.append(nn.Conv2d(in_channels, hidden_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(hidden_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_channels = hidden_channels

        # Final convolution to get a single-channel attention map
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=kernel_size, padding=padding, bias=False))
        self.conv = nn.Sequential(*layers)
        
        # Use LeakyReLU instead of Tanh to avoid saturation and allow stronger gradients.
        self.act = nn.LeakyReLU(0.2, inplace=True)
        
        # Increase the initial scaling factor to allow stronger attention signals.
        self.alpha = nn.Parameter(torch.tensor(1.0))
        
        # Weight initialization for each Conv2d layer
        for layer in self.conv:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_uniform_(layer.weight, a=0.2, nonlinearity='leaky_relu')

    def forward(self, x):
        # x: (B, C, H, W)
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)
        x_cat = torch.cat([avg_pool, max_pool], dim=1)     # (B, 2, H, W)
        attn = self.conv(x_cat)
        attn = self.act(attn)
        out = x + self.alpha * attn
        return out


class CrossModalProjection(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.face_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model*2),
            nn.GELU(),
            nn.Linear(d_model*2, d_model)
        )
        self.audio_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model*2),
            nn.GELU(),
            nn.Linear(d_model*2, d_model)
        )
        
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

class DenseBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels*2, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels*3, in_channels, 3, padding=1)  # 包含跳跃连接
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat([x, x1], dim=1))
        return x + x2

class DynamicFusion(nn.Module):
    def __init__(self, num_modalities):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_modalities))
        
    def forward(self, features):
        norm_weights = F.softmax(self.weights, dim=0)
        return sum(w * f for w, f in zip(norm_weights, features))

class MultiScaleFusion(nn.Module):
    def __init__(self, in_channels_list, out_channels, target_shape):
        """
        Args:
            in_channels_list (list[int]): List of channel numbers for each feature map to fuse.
            out_channels (int): The desired number of channels for the fused feature.
            target_shape (tuple): (H, W) target spatial resolution for each feature.
        """
        super(MultiScaleFusion, self).__init__()
        self.target_shape = target_shape
        # Project each input feature map to the common out_channels via 1x1 convolutions.
        self.proj_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1)
            for in_ch in in_channels_list
        ])
        
        self.dynamic_fusion = DynamicFusion(2)
        
        # Fuse the concatenated features with a 3x3 convolution.
        self.fusion_conv = nn.Conv2d(out_channels * len(in_channels_list), out_channels, kernel_size=3, padding=1)
        
    def forward(self, features):
        # Adaptive pool each feature to the target shape.
        pooled_feats = [F.adaptive_avg_pool2d(f, self.target_shape) for f in features]
        # Project each pooled feature.
        projected_feats = [conv(feat) for conv, feat in zip(self.proj_convs, pooled_feats)]
        # Concatenate along the channel dimension.
        #concatenated = torch.cat(projected_feats, dim=1)
        
        fused = self.dynamic_fusion(projected_feats)
        
        # Fuse via a convolution.
        #fused = self.fusion_conv(fused)
        return fused

class TransformerSyncnet(nn.Module):
    def __init__(self, num_heads=8, num_encoder_layers=4, embed_dim=512):
        super(TransformerSyncnet, self).__init__()
        
        self.embed_dim = embed_dim
        # --- Face encoder for individual frames ---
        self.face_encoder1 = nn.Sequential(
            # Input: (B, 15, H, W)  where 15 = 5 images x 3 channels
            Conv2d(15, 128, kernel_size=3, stride=2, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            #MouthAttention(128),
            SpatialAttention(hidden_channels=128)
        )
        
        self.face_encoder2 = nn.Sequential(
            Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # Downsample
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            #MouthAttention(256),
            SpatialAttention(hidden_channels=256)
        )
        
        self.face_encoder3 = nn.Sequential(
            Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # Downsample width
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            #MouthAttention(512),
            SpatialAttention(hidden_channels=512)
        )
        
        self.face_encoder4 = nn.Sequential(
            Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),  # Downsample
            Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, residual=True),
        )
        
        # self.face1_to_face3_skip = nn.Sequential(
        #     # Input: (B, 15, H, W)  where 15 = 5 images x 3 channels
        #     Conv2d(128, 512, kernel_size=3, stride=2, padding=1),
        #     Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), 
        # )
                
        # --- Audio encoder ---
        self.audio_encoder1 = nn.Sequential(
            # Example input shape: (B, 1, H_audio, W_audio)
            Conv2d(1, 64, kernel_size=3, stride=1, padding=1, leaking=0.05),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            
        )

        self.audio_encoder2 = nn.Sequential(
            Conv2d(64, 128, kernel_size=3, stride=1, padding=1, leaking=0.05),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            
        )
        
        self.audio_encoder3 = nn.Sequential(
            Conv2d(128, 256, kernel_size=3, stride=1, padding=1, leaking=0.05),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            
        )
        
        # Additional layer for matching dimensions if necessary
        self.audio_skip = Conv2d(64, 256, kernel_size=3, stride=1, padding=1)
        self.audio_encoder4 = nn.Sequential(
            Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
        )        
        
        #self.cross_proj = CrossModalProjection(1024)
        
        target_shape = (48, 96)  # (20, 24) in this case
        
        # --- Multi-Scale Fusion for face features ---
        # Here we fuse features from face_encoder2 (256 channels), face_encoder3 (512 channels), and face_encoder4 (1024 channels)
        self.face_multi_scale_fusion = MultiScaleFusion(in_channels_list=[256, 512, 1024], out_channels=1024, target_shape=target_shape)
        self.audio_multi_scale_fusion = MultiScaleFusion(in_channels_list=[128, 256, 1024], out_channels=1024, target_shape=target_shape)
        
        
        self.pos_encoder = PositionalEncoding2D(1024, 48, 96)

        self.ln1 = nn.LayerNorm(1024)
        self.ln2 = nn.LayerNorm(1024)
        
        
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(1024, num_heads, dropout=0.1) 
            for _ in range(num_encoder_layers)
        ])

        
        # Final classification head.
        # We pool tokens for each modality separately, then concatenate their global features.
        self.classifier = nn.Sequential(
            nn.Linear(1024, 64),
            nn.LeakyReLU(0.01, inplace=False),
            nn.Dropout(p=0.3),
            nn.Linear(64, 1)  # binary classification output
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

        # Calculate min and max values
        min_value = torch.min(audio_embedding)
        max_value = torch.max(audio_embedding)

        epsilon = 1e-6

        # Normalize the tensor to range (0, 1)
        audio_embedding = (audio_embedding - min_value) / (max_value - min_value + epsilon)

        # Scale to ensure it does not reach exactly 0 or 1
        audio_embedding = audio_embedding * (1 - epsilon) + epsilon
        
        save_every_s_steps = 2000
        
        face1 = self.face_encoder1(face_embedding)
        #face1_to_face3 = self.face1_to_face3_skip(face1)
        
        if step % save_every_s_steps == 0:
          self.save_sample_images(face1, 'face1', step)
        
        face2 = self.face_encoder2(face1)
        if step % save_every_s_steps == 0:
          self.save_sample_images(face2, 'face2', step)
        
        #face2 = face2 + face_skip1
                        
        face3 = self.face_encoder3(face2)
        if step % save_every_s_steps == 0:
          self.save_sample_images(face3, 'face3', step)
        #face3 = face3 + face1_to_face3
        
        face4 = self.face_encoder4(face3)
        
        if step % save_every_s_steps == 0:
          self.save_sample_images(face4, 'face4', step)
        
        # Fuse multi-scale face features.
        fused_face_features = self.face_multi_scale_fusion([face2, face3, face4])
        face_features = fused_face_features
        
        # --- Process audio modality ---
        audio_features1 = self.audio_encoder1(audio_embedding)  # (B, 512, H_a, W_a)
        
        audio1_to_4_skip = self.audio_skip(audio_features1)
        
        
        if step % save_every_s_steps == 0:
          self.save_sample_images(audio_features1, 'audio1', step)
        
        audio_features2 = self.audio_encoder2(audio_features1)
        
        audio_features3 = self.audio_encoder3(audio_features2)
        
        audio_features4 = self.audio_encoder4(audio_features3 + audio1_to_4_skip)
        
        fused_audio_features = self.audio_multi_scale_fusion([audio_features2, audio_features3, audio_features4])
        
        audio_features = fused_audio_features
        
        # --- Apply positional encoding separately to each modality ---
        face_features = self.pos_encoder(face_features)
        audio_features = self.pos_encoder(audio_features)
                
        
        B, C, H, W = face_features.shape
        # Flatten spatial dimensions: (B, C, H, W) -> (B, C, H*W)
        face_flat = face_features.view(B, C, -1)  # shape: (B, C, S)
        audio_flat = audio_features.view(B, C, -1)  # shape: (B, C, S)
        
        # Permute to (S, B, C) as required by nn.MultiheadAttention
        face_seq = face_flat.permute(2, 0, 1)   # (S, B, C)
        audio_seq = audio_flat.permute(2, 0, 1)   # (S, B, C)
        
        face_seq = self.ln1(face_seq)
        
        attn_output = face_seq

        # Let face features (lips) attend to audio features:
        # face_seq acts as the query, and audio_seq provides key and value.
        for layer in self.attn_layers:
          attn_output = layer(attn_output, audio_seq, audio_seq)[0]        

        
        # Reshape back to (B, C, H, W)
        attn_output = attn_output.permute(1, 2, 0).view(B, C, H, W)
        
        attn_output = attn_output + face_features
        
        pooled = nn.AdaptiveAvgPool2d((1, 1))(attn_output)
        flattened = pooled.view(pooled.size(0), -1)
        
        result = self.classifier(flattened)
        
        if step % save_every_s_steps == 0:
          self.save_sample_images(face_features, 'face_final', step)
          self.save_sample_images(audio_features, 'audio_final', step)
          self.save_sample_images(attn_output, 'final_fused', step)
          self.save_sample_images(fused_audio_features, 'audio_fused', step)
        
        return result, attn_output, fused_audio_features

    def save_sample_images(self, x, layer, step):
        base_dir = 'temp'
        B, C, H, W = x.shape
        for b in range(B):
          if b > 2:
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
          filename = os.path.join(base_dir, f"{layer}_{step}_sample_{b}_combined.png")
          cv2.imwrite(filename, image_bgr)
        
        
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
        