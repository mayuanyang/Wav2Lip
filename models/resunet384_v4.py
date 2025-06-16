import torch
import torch.nn as nn
import torchvision.models as models

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
                
        
        # 定义解码器块
        self.decoder4 = DecoderBlock(in_channels=384, skip_channels=136, out_channels=512)
        self.decoder3 = DecoderBlock(in_channels=520, skip_channels=40, out_channels=256)
        self.decoder2 = DecoderBlock(in_channels=256, skip_channels=24, out_channels=128, scale_factor=4)
        self.decoder1 = DecoderBlock(in_channels=128, skip_channels=0, out_channels=64)
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, num_classes, kernel_size=1),
            nn.Tanh()  # 取决于任务需求
        )
    
    def forward(self, audio_sequences, face_sequences, step):
        B = audio_sequences.size(0)       
        input_dim_size = len(face_sequences.size())
        
        if input_dim_size > 4:
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)

        expanded_B = face_sequences.size(0)  # 展平后的 batch size
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

