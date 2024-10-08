import torch
import torch.nn as nn

class LoRAConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, lora_rank=4, lora_scaling=1.0):
        super(LoRAConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, dilation, groups, bias)
        self.lora_A = nn.Parameter(torch.zeros(lora_rank, in_channels * kernel_size[0] * kernel_size[1]))
        self.lora_B = nn.Parameter(torch.zeros(out_channels, lora_rank))
        nn.init.normal_(self.lora_A, std=0.02)
        nn.init.zeros_(self.lora_B)
        self.lora_scaling = lora_scaling  # You can adjust this scaling factor

    def forward(self, x):
        original_output = self.conv(x)
        # Compute LoRA adjustment
        weight = self.conv.weight.view(self.conv.out_channels, -1)

        # Adjusting lora_A to match kernel dimensions
        lora_adjustment = torch.matmul(self.lora_B, self.lora_A).view(self.conv.out_channels, *self.conv.weight.shape[1:])

        # Adjusting the LoRA weight
        adjusted_weight = self.conv.weight + self.lora_scaling * lora_adjustment

        # Perform convolution with adjusted weights
        output = nn.functional.conv2d(
            x, adjusted_weight, self.conv.bias, self.conv.stride,
            self.conv.padding, self.conv.dilation, self.conv.groups)

        return output
    
class LoRATransposeConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 lora_rank=4, lora_scaling=1.0):
        super(LoRATransposeConv2d, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride,
                                       padding)
        
        # Handle both integer and tuple kernel sizes
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        # LoRA parameters
        self.lora_A = nn.Parameter(torch.zeros(lora_rank, in_channels * kernel_size[0] * kernel_size[1]))
        self.lora_B = nn.Parameter(torch.zeros(out_channels, lora_rank))
        
        nn.init.normal_(self.lora_A, std=0.02)
        nn.init.zeros_(self.lora_B)
        
        self.lora_scaling = lora_scaling  # Scaling factor for LoRA adjustment

    def forward(self, x):
        # Compute LoRA adjustment
        weight_flat = self.conv.weight.view(self.conv.out_channels, -1)
        lora_adjustment = torch.matmul(self.lora_B, self.lora_A).view_as(self.conv.weight)
        
        # Adjusted weight with LoRA scaling
        adjusted_weight = weight_flat + self.lora_scaling * lora_adjustment.view(weight_flat.size())
        
        # Perform convolution with adjusted weights
        output = nn.functional.conv_transpose2d(
            x, adjusted_weight.view_as(self.conv.weight), self.conv.bias, 
            stride=self.conv.stride, padding=self.conv.padding, 
            output_padding=self.conv.output_padding, dilation=self.conv.dilation, 
            groups=self.conv.groups
        )

        return output

