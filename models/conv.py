import torch
from torch import nn
from torch.nn import functional as F

class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout) # Eddy, looks like if the dataset is too small, it will cause error, see this link https://discuss.pytorch.org/t/error-expected-more-than-1-value-per-channel-when-training/26274
                            )
        self.act = nn.LeakyReLU(0.01, inplace=True)
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)

class nonorm_Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            )
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)

class Conv2dTranspose(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, output_padding=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.ConvTranspose2d(cin, cout, kernel_size, stride, padding, output_padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        # Kernel size must be odd for padding to keep the output size same as input
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        
        # Convolutional layer for spatial attention
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Max pool and average pool across the channel dimension to reduce to 2 channels
        print('The x shape', x.shape)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # Max pooling over channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)    # Average pooling over channel dimension
        
        # Concatenate along the channel axis (C = 2, from max pool and avg pool)
        attention_map = torch.cat([max_out, avg_out], dim=1)
        
        # Apply the spatial attention convolution
        attention_map = self.conv(attention_map)
        
        # Apply sigmoid to get attention weights in the range [0, 1]
        attention_map = self.sigmoid(attention_map)
        
        # Multiply input with spatial attention map
        return x * attention_map
