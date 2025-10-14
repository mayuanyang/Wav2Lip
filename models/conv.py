import torch
from torch import nn
from torch.nn import functional as F

class ZeroSensitiveBatchNorm2d(nn.Module):
    def __init__(self, num_features):
        super(ZeroSensitiveBatchNorm2d, self).__init__()
        self.bn = nn.BatchNorm2d(num_features)

    def forward(self, input):
        # 记录哪些位置是0
        zero_mask = input == 0

        # 对输入进行归一化
        output = self.bn(input)

        # 将原本为0的位置重新设为0
        output[zero_mask] = 0

        return output

# 使用自定义的BatchNorm2D替换原有的BatchNorm2D
class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, dilation=1, leaking=0.01, use_zero_sensitive=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding, dilation=dilation),
                            ZeroSensitiveBatchNorm2d(cout) if use_zero_sensitive else nn.BatchNorm2d(cout)
                            )
        self.act = nn.GELU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out = out + x
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
