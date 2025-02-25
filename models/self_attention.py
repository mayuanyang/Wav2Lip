import torch
import math
from torch import nn
from torch.nn import functional as F

from .cross_modal_attention import CrossModalAttention2d


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, sparse_attention=False, reduction=8, window_size=9):
        """
        Args:
            in_channels: Number of input channels.
            sparse_attention: If True, use fixed window-based attention.
            reduction: Factor to reduce channels for query and key.
            window_size: Fixed number of neighboring pixels for attention. Must be a perfect square.
        """
        super(AttentionBlock, self).__init__()
        self.sparse_attention = sparse_attention
        
        # Ensure window_size is a perfect square (e.g. 9, 16, 25, etc.)
        sqrt_ws = math.sqrt(window_size)
        if sqrt_ws != int(sqrt_ws):
            raise ValueError("window_size must be a perfect square")
        self.kernel_size = int(sqrt_ws)
        self.padding = self.kernel_size // 2
        
        self.query = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.key   = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        
        if self.sparse_attention:
            # Compute query, key, value feature maps
            q = self.query(x)  # shape: [B, C_reduced, H, W]
            k = self.key(x)    # shape: [B, C_reduced, H, W]
            v = self.value(x)  # shape: [B, C, H, W]
            
            # Unfold key and value to extract fixed window neighborhoods
            k_unfold = F.unfold(k, kernel_size=self.kernel_size, padding=self.padding)  
            v_unfold = F.unfold(v, kernel_size=self.kernel_size, padding=self.padding)
            # k_unfold: [B, C_reduced * (kernel_size^2), H*W]
            # Reshape to [B, H*W, kernel_size^2, C_reduced]
            K = self.kernel_size * self.kernel_size
            k_unfold = k_unfold.view(B, -1, K, H * W).permute(0, 3, 2, 1)
            # v_unfold: [B, C * (kernel_size^2), H*W] -> [B, H*W, kernel_size^2, C]
            v_unfold = v_unfold.view(B, -1, K, H * W).permute(0, 3, 2, 1)
            
            # Reshape query to [B, H*W, C_reduced]
            q_flat = q.view(B, q.size(1), H * W).permute(0, 2, 1)
            
            # Compute dot-product attention within each fixed window
            # Normalize query and key tensors
            q_flat = F.normalize(q_flat, p=2, dim=-1)  # L2 normalize along the last dimension
            k_unfold = F.normalize(k_unfold, p=2, dim=-1)  # L2 normalize along the last dimension

            energy = torch.einsum('bqc, bqkc -> bqk', q_flat, k_unfold)
            energy = energy - energy.max(dim=-1, keepdim=True)[0]
            if torch.min(energy) < -50 or torch.max(energy) > 50:
              print(f"Min and Max energy: {torch.min(energy)}. {torch.max(energy)}")
            
            attn = F.softmax(energy, dim=-1)
            
            # Aggregate values
            out = torch.einsum('bqk, bqkc -> bqc', attn, v_unfold)
            out = out.permute(0, 2, 1).view(B, C, H, W)
        else:
            # Full attention over all pixels
            q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
            k = self.key(x).view(B, -1, H * W).permute(0, 2, 1)
            v = self.value(x).view(B, -1, H * W).permute(0, 2, 1)

            # L2 normalize query and key
            q = F.normalize(q, p=2, dim=-1)  # Normalize along the last dimension (C_reduced)
            k = F.normalize(k, p=2, dim=-1)  # Normalize along the last dimension (C_reduced)
            
            q = q.unsqueeze(1)
            k = k.unsqueeze(1)
            v = v.unsqueeze(1)
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.1)
            out = out.squeeze(1)
            out = out.permute(0, 2, 1).view(B, C, H, W)
            
        return self.gamma * out + x