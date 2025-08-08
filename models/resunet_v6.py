import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import GaussianBlur


from .conv import Conv2dTranspose, Conv2d, nonorm_Conv2d

def linear_schedule(steps=200):
    """Generate a linear schedule for diffusion steps"""
    # Create a linear schedule from 0.0001 to 0.02 for 20 steps
    return torch.linspace(0.1, 0.1, steps)
  
class ResUNet384V6(nn.Module):
    def __init__(self, num_of_blocks=2, diffusion_steps=1):
        super(ResUNet384V6, self).__init__()
        
        self.diffusion_steps = diffusion_steps
        self.betas = linear_schedule(diffusion_steps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.ellipse_params = {
            'center': (0.0, 0),  # (x,y)中心偏移(归一化坐标)
            'axes': (1, 0.75),      # (宽,高)比例
            'blur': 0.0             # 边缘模糊系数(相对于短边)
        }
        
        self.blocks = nn.ModuleList()
        
        
        for i in range(num_of_blocks):
            self.blocks.append(ProcessBlock384(3))
        
        print('The length of blocks', len(self.blocks))

        self.bn12 = nn.BatchNorm2d(12)

        self.bn3 = nn.BatchNorm2d(3)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
        
    def freeze_blocks(self, num_blocks_to_freeze):
        """
        Freeze the parameters of the first num_blocks_to_freeze blocks
        
        Args:
            num_blocks_to_freeze (int): Number of blocks to freeze
        """
        for i in range(min(num_blocks_to_freeze, len(self.blocks))):
            for param in self.blocks[i].parameters():
                param.requires_grad = False
                print('freezeing')
                
    def unfreeze_blocks(self, num_blocks_to_unfreeze=None):
        """
        Unfreeze the parameters of the blocks
        
        Args:
            num_blocks_to_unfreeze (int, optional): Number of blocks to unfreeze.
                If None, unfreeze all blocks.
        """
        if num_blocks_to_unfreeze is None:
            num_blocks_to_unfreeze = len(self.blocks)
            
        for i in range(min(num_blocks_to_unfreeze, len(self.blocks))):
            for param in self.blocks[i].parameters():
                param.requires_grad = True
    
    def generate_ellipse_mask(self, h, w, split_idx, device):
        """生成下半部分的椭圆遮罩"""
        y_bottom = torch.linspace(0, 1, h - split_idx, device=device) * 2 - 1
        x_coord = torch.linspace(-1, 1, w, device=device)
        y_grid, x_grid = torch.meshgrid(y_bottom, x_coord, indexing='ij')
        
        dx, dy = self.ellipse_params['center']
        a_ratio, b_ratio = self.ellipse_params['axes']
        
        adjusted_a = a_ratio * (w / h)  
        
        ellipse_mask = ((x_grid - dx)/adjusted_a)**2 + \
                      ((y_grid - dy)/b_ratio)**2 <= 1.0
        
        mask = ellipse_mask.float()
        if self.ellipse_params['blur'] > 0:
            kernel_size = int(min(h, w) * self.ellipse_params['blur']) | 1
            gaussian_blur_op = lambda x: GaussianBlur(kernel_size=(kernel_size, kernel_size), sigma=kernel_size/3)(x)
            mask = gaussian_blur_op(mask.unsqueeze(0).unsqueeze(0)).squeeze()
        
        return mask
      
    def diffuse(self, x, t, channels_to_mask=3):
        """
        带椭圆遮罩的扩散过程
        """
        b, c, h, w = x.shape
        split_idx = h // 2
        top_half = x[:, :, :split_idx, :]
        bottom_half = x[:, :, split_idx:, :]
        
        rgb_channels = bottom_half[:, :channels_to_mask, :, :]
        other_channels = bottom_half[:, channels_to_mask:, :, :]
        
        mask = self.generate_ellipse_mask(h, w, split_idx, x.device)
        mask = mask.unsqueeze(0).unsqueeze(0).repeat(b, channels_to_mask, 1, 1)
        
        sqrt_alpha_t = torch.sqrt(self.alphas_cumprod[t]).view(b, 1, 1, 1)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - self.alphas_cumprod[t]).view(b, 1, 1, 1)
        
        epsilon = torch.randn_like(rgb_channels)
        noisy_rgb = sqrt_alpha_t * rgb_channels + sqrt_one_minus_alpha_t * epsilon
        
        noisy_rgb = rgb_channels * (1 - mask) + noisy_rgb * mask
        
        noisy_bottom = torch.cat([noisy_rgb, other_channels], dim=1)
        result = torch.cat([top_half, noisy_bottom], dim=2)

        return result
      
    def sample_t(self, expanded_B, probabilities=None):
        """Sample timesteps for diffusion process"""
        if probabilities is None:
            # Uniform distribution over all steps
            probabilities = torch.ones(self.diffusion_steps, dtype=torch.float32)
        else:
            # Ensure probabilities match the number of steps
            assert len(probabilities) == self.diffusion_steps, "Probabilities must match the number of diffusion steps"
        
        probabilities = probabilities / probabilities.sum()
        t = torch.multinomial(probabilities, expanded_B, replacement=True)
        return t


    def forward(self, audio_sequences, face_sequences, place = None, step=None):
        temp_output = None
        
        input_dim_size = len(face_sequences.size())
        B = audio_sequences.size(0)       
        
        if input_dim_size > 4:
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)

        expanded_B = face_sequences.size(0)
        
        # Handle diffusion steps
        if step is None:
            # Randomly sample a diffusion step for training (standard diffusion training mode)
            t0 = torch.randint(0, self.diffusion_steps, (expanded_B,), device=face_sequences.device).long()
        else:
            # Use fixed timestep if provided (for backward compatibility)
            t0 = torch.full((expanded_B,), step % self.diffusion_steps, dtype=torch.long, device=face_sequences.device)
        
        self.alphas_cumprod = self.alphas_cumprod.to(face_sequences.device)
        noisy_face_sequences = self.diffuse(face_sequences.float(), t0, 3)

        for i, block in enumerate(self.blocks):
            if temp_output is not None:
                new_input = torch.cat([temp_output, face_sequences[:, 3:, :, :]], dim=1)  # 第二个及后续块处理前一个块的输出
                face_input = new_input
            else:
                face_input = noisy_face_sequences  # 第一个块处理噪声图像

            temp_output = self.forward_impl(audio_sequences, face_input, block.face_encoder_blocks, block.audio_encoder, block.face_decoder_blocks, block.output_block)
        
        outputs = temp_output
        if input_dim_size > 4:
            outputs = torch.split(temp_output, B, dim=0) # [(B, C, H, W)]
            outputs = torch.stack(outputs, dim=2) # (B, C, T, H, W)
      
        return outputs, None, None

    def forward_impl(self, audio_sequences, face_sequences, face_encoder_blocks, audio_encoder, face_decoder_blocks, output_block):
        # audio_sequences = (B, T, 1, 80, 16)
        B = audio_sequences.size(0)

        input_dim_size = len(face_sequences.size())
        if input_dim_size > 4:
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)

        audio_embedding = audio_encoder(audio_sequences) # B, 512, 1, 1

        face_features = []
        this_face_sequence = face_sequences
        for f in face_encoder_blocks:
            this_face_sequence = f(this_face_sequence)
            face_features.append(this_face_sequence)

        
        x = audio_embedding
        # Apply attention before concatenation
        for i, f in enumerate(face_decoder_blocks):
            x = f(x)
            if face_features:
                skip = face_features.pop()
                x = torch.cat((x, skip), dim=1)

        x = output_block(x)
                    
        return x


class ProcessBlock384(nn.Module):
    def __init__(self, output_block_channels) -> None:
        super(ProcessBlock384, self).__init__()
        '''
        Outpu = Input + (k-1) x S

        Where:
        Input is the receptive field size from the previous layer.
        k is the kernel size.
        S is the stride.
        '''

        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(12, 64, kernel_size=7, stride=1, padding=3), #1+(7−1)×1=7
                          Conv2d(64, 64, kernel_size=7, stride=1, padding=3, residual=True), #7+(7-1)*1=13
                          Conv2d(64, 64, kernel_size=7, stride=1, padding=3, residual=True), #13+(7-1)*1=19
                          ), # 384,384
            
            nn.Sequential(Conv2d(64, 192, kernel_size=3, stride=2, padding=1), #13+(7−1)×2=25
                          Conv2d(192, 192, kernel_size=3, stride=1, padding=1, residual=True), #25+(7-1)*1=31
                          Conv2d(192, 192, kernel_size=3, stride=1, padding=1, residual=True), #25+(7-1)*1=31
                          ), # 192,192

            nn.Sequential(Conv2d(192, 192, kernel_size=3, stride=2, padding=1), #31+(7−1)×2=43
              Conv2d(192, 192, kernel_size=3, stride=1, padding=1, residual=True), #49
              Conv2d(192, 192, kernel_size=3, stride=1, padding=1, residual=True), #49
              ), # 96,96

            nn.Sequential(Conv2d(192, 192, kernel_size=3, stride=2, padding=1), # 48,48, 49+(7−1)×2=61
            Conv2d(192, 192, kernel_size=3, stride=1, padding=1, residual=True), #67
            ),

            nn.Sequential(Conv2d(192, 192, kernel_size=3, stride=2, padding=1), # 24,24, 67+(7−1)×2=79
            Conv2d(192, 192, kernel_size=3, stride=1, padding=1, residual=True), #85
            ),

            nn.Sequential(Conv2d(192, 192, kernel_size=3, stride=2, padding=1), # 12,12, 85+(7−1)×2=97
            Conv2d(192, 192, kernel_size=3, stride=1, padding=1, residual=True), # 103
            ), 

            nn.Sequential(Conv2d(192, 192, kernel_size=5, stride=2, padding=2), # 6,6, 103+(5−1)×2=111
            Conv2d(192, 192, kernel_size=5, stride=1, padding=2, residual=True) # 115
            ), 

            nn.Sequential(Conv2d(192, 384, kernel_size=5, stride=2, padding=2), # 3,3, 115+(5−1)×2=123
            Conv2d(384, 384, kernel_size=5, stride=1, padding=2, residual=True), # 127
            ), 
            
            ]) # 45

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0, residual=True),)

        self.face_decoder_blocks = nn.ModuleList([
            

            nn.Sequential(Conv2dTranspose(512, 256, kernel_size=3, stride=1, padding=0), # 3,3
            Conv2d(256, 256, kernel_size=5, stride=1, padding=2, residual=True),),

            nn.Sequential(Conv2dTranspose(640, 320, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(320, 320, kernel_size=5, stride=1, padding=2, residual=True),), # 6, 6

            nn.Sequential(Conv2dTranspose(512, 352, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(352, 352, kernel_size=5, stride=1, padding=2, residual=True),
            ), # 12, 12

            nn.Sequential(Conv2dTranspose(544, 368, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(368, 368, kernel_size=5, stride=1, padding=2, residual=True),
            ), # 24, 24

            nn.Sequential(Conv2dTranspose(560, 376, kernel_size=3, stride=2, padding=1, output_padding=1), 
            Conv2d(376, 376, kernel_size=5, stride=1, padding=2, residual=True),
            ), # 48, 48

            nn.Sequential(Conv2dTranspose(568, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(128, 128, kernel_size=5, stride=1, padding=2, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            ), # 96,96
            
            nn.Sequential(
                Conv2dTranspose(320, 160, kernel_size=3, stride=2, padding=1, output_padding=1),
                Conv2d(160, 160, kernel_size=5, stride=1, padding=2, residual=True),
                Conv2d(160, 160, kernel_size=3, stride=1, padding=1, residual=True),
            ),
            nn.Sequential(
                Conv2dTranspose(352, 112, kernel_size=3, stride=2, padding=1, output_padding=1),
                Conv2d(112, 112, kernel_size=5, stride=1, padding=2, residual=True),
                Conv2d(112, 112, kernel_size=3, stride=1, padding=1, residual=True),
            )
            ]) 

        self.output_block = nn.Sequential(
          nn.Conv2d(176, output_block_channels, kernel_size=1, stride=1, padding=0),
          nn.Sigmoid())
