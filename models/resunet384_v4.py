import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import GaussianBlur

def modify_efficientnet_conv1(effnet, in_channels):
    """
    Modify the first convolutional layer of EfficientNet to accept `in_channels` input channels.

    Args:
        effnet (nn.Module): The EfficientNet model to modify.
        in_channels (int): The number of input channels.

    Returns:
        nn.Module: The modified convolutional layer.
    """
    old_conv = effnet.features[0][0]  # Accessing the first Conv2d layer
    new_conv = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None
    )

    with torch.no_grad():
        if in_channels == 3:
            new_conv.weight.copy_(old_conv.weight)
        elif in_channels > 3:
            # Initialize the first 3 channels with the pretrained weights
            new_conv.weight[:, :3, :, :] = old_conv.weight
            # Initialize the remaining channels by taking the mean of the pretrained weights
            mean_weight = old_conv.weight.mean(dim=1, keepdim=True)
            for i in range(3, in_channels):
                new_conv.weight[:, i:i+1, :, :] = mean_weight
        else:
            # If in_channels < 3, copy the first `in_channels` weights
            new_conv.weight.copy_(old_conv.weight[:, :in_channels, :, :])

    return new_conv

def linear_schedule():
    return torch.tensor([0.6, 0.7, 0.8, 0.9])
  
class EfficientNetEncoder(nn.Module):
    def __init__(self, model_name='efficientnet-b0'):
        super(EfficientNetEncoder, self).__init__()
        self.model = models.efficientnet_b3(pretrained=True)
        #print(self.model)
        self.model.features[0][0] = modify_efficientnet_conv1(self.model, in_channels=12)
        
        # 提取不同阶段的特征
        self.stage1 = nn.Sequential(*self.model.features[:2])   # 输出尺寸较大
        self.stage2 = nn.Sequential(*self.model.features[2:4])
        self.stage3 = nn.Sequential(*self.model.features[4:6])
        self.stage4 = nn.Sequential(*self.model.features[6:8])  # 最终输出 (batch, 1536, 12, 12)
        
        
        
    def forward(self, x):
        features = []
        # x = self.model.conv_stem(x)
        # x = self.model.bn1(x)
        # x = self.model.act1(x)
        
        x = self.stage1(x)
        features.append(x)  # 例如 (batch, 32, 112, 112)
        x = self.stage2(x)
        features.append(x)  # 例如 (batch, 24, 56, 56)
        x = self.stage3(x)
        features.append(x)  # 例如 (batch, 40, 28, 28)
        x = self.stage4(x)
        features.append(x)  # (batch, 1536, 12, 12)
        
        return features

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, scale_factor=2):
        super(DecoderBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, skip):
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            
        x = self.conv(x)
        return x

class ResUNet384V4(nn.Module):
    def __init__(self, num_classes=3, model_name='efficientnet-b0'):
        super(ResUNet384V4, self).__init__()
        self.encoder = EfficientNetEncoder()
        
        self.betas = linear_schedule()  # or cosine_noise_schedule()
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.ellipse_params = {
            'center': (0.0, 0),  # (x,y)中心偏移(归一化坐标)
            'axes': (1, 0.75),      # (宽,高)比例
            'blur': 0.03             # 边缘模糊系数(相对于短边)
        }
                
        
        # 定义解码器块
        self.decoder4 = DecoderBlock(in_channels=384, skip_channels=136, out_channels=512)
        self.decoder3 = DecoderBlock(in_channels=520, skip_channels=40, out_channels=256)
        self.decoder2 = DecoderBlock(in_channels=256, skip_channels=24, out_channels=128, scale_factor=4)
        self.decoder1 = DecoderBlock(in_channels=128, skip_channels=0, out_channels=64)
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, num_classes, kernel_size=1),
            nn.Tanh()  # 取决于任务需求
        )
        
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
      
    def forward(self, audio_sequences, face_sequences, step):
        B = audio_sequences.size(0)       
        input_dim_size = len(face_sequences.size())
        
        if input_dim_size > 4:
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)

        expanded_B = face_sequences.size(0)  # 展平后的 batch size
        
        t0 = self.sample_t(expanded_B, torch.tensor([1, 1, 1, 1], dtype=torch.float32)).to(face_sequences.device)
       
        self.alphas_cumprod = self.alphas_cumprod.to(face_sequences.device)
        
        face_sequences = self.diffuse(face_sequences.float(), t0, 3)
        
        features = self.encoder(face_sequences)
        f1, f2, f3, f4 = features  # 假设顺序
        
        d4 = self.decoder4(f4, f3)  # (B, 512, 24, 24)
        
        d3 = self.decoder3(d4, f2)  # (B, 256, 48, 48)
        
        d2 = self.decoder2(d3, f1)  # (B, 128, 192, 192)
        
        d1 = self.decoder1(d2, None)  # (B, 64, 192, 192)
        
        out = self.final_conv(d1)  # (B, num_classes, 192, 192)
        
        if input_dim_size > 4:
            out = torch.split(out, B, dim=0)
            out = torch.stack(out, dim=2)
        
        return out, None, None

