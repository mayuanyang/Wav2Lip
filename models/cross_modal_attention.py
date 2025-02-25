import torch
from torch import nn
from torch.nn import functional as F

from .conv import Conv2dTranspose, Conv2d, nonorm_Conv2d

class CrossModalAttention2d(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(CrossModalAttention2d, self).__init__()
        # Project face features into a lower-dimensional query space
        self.query_conv = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        # Project audio features into key space
        self.key_conv = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        # Project audio features into value space (we keep full channel dimension)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        # Learnable scaling factor, init to 0, but will learn as it goes
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, face_feat, audio_feat):
        """
        face_feat: Tensor of shape [B, C, H, W] from the face bottleneck
        audio_feat: Tensor of shape [B, C, H, W] from the audio encoder
        """
        FB, FC, FH, FW = face_feat.size()
        AB, AC, AH, AW = audio_feat.size()
        # Compute query from face features
        query = self.query_conv(face_feat).view(FB, -1, FH * FW).permute(0, 2, 1)  # (B, HW, C//reduction)
        
        # Compute key from audio features
        key = self.key_conv(audio_feat).view(AB, -1, AH * AW)                       # (B, C//reduction, HW)

        query = F.normalize(query, p=2, dim=-1)  # L2 normalize along the last dimension
        key = F.normalize(key, p=2, dim=-1)  # L2 normalize along the last dimension
        
        energy = torch.bmm(query, key)
        energy = energy - energy.max(dim=-1, keepdim=True)[0]
        #print(f"Min and Max energy: {torch.min(energy)}. {torch.max(energy)}")
        
        #energy = energy.clamp(min=-50, max=50)
        attention = F.softmax(energy, dim=-1)
        
        # Compute value from audio features
        value = self.value_conv(audio_feat).view(AB, -1, AH * AW)                   # (B, C, HW)
        
        # Aggregate audio features using the attention map
        out = torch.bmm(value, attention.permute(0, 2, 1))                       # (B, C, HW)
        
        out = out.view(FB, FC, FH, FW)
        # Residual connection
        out = self.gamma * out + face_feat
        return out
