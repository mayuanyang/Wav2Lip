import torch
import torch.nn as nn
import torch.nn.functional as F

class ResUNet384V8(nn.Module):
    def __init__(self, print_gradients=False):
        super(ResUNet384V8, self).__init__()
        self.print_gradients = print_gradients
        
        # Enhanced configuration
        self.ellipse_params = {
            'center': (0.0, 0),
            'axes': (1, 0.75),
            'blur': 0.0
        }
        
        # === Improved Encoder Blocks ===
        self.bottom_encoder = BottomEncoder()
        self.audio_encoder = AudioEncoder()
        
        # === Bottleneck with Attention ===
        self.bottleneck = CrossModalBottleneck()
        
        # === Improved Decoder ===
        self.bottom_decoder = BottomDecoder()
        
        # === Output Layers ===
        self.output_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid()
        )
        
        self.enhancer_encoder = BottomEncoder()
        self.enhancer_audio_encoder = AudioEncoder()
        
        # === Bottleneck with Attention ===
        self.enhancer_bottleneck = CrossModalBottleneck()
        
        # === Improved Decoder ===
        self.enhancer_decoder = BottomDecoder()
        
        # === Output Layers ===
        self.enhancer_output_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid()
        )
        
        # Gradient monitoring
        self.gradient_hooks = []
        
    def forward(self, audio_sequences, face_sequences, step=None, training=True, train_enhancer=False):
        # Handle sequence inputs
        input_dim_size = len(face_sequences.size())
        B = audio_sequences.size(0)
        
        if input_dim_size > 4:
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)
        
        # Store original top half
        original_top_half = face_sequences[:, :3, :192, :]
        
        # Apply diffusion
        face_sequences_3ch = self.diffuse(face_sequences[:, :3, :, :], channels_to_mask=3)
        face_sequences_ref = face_sequences[:, 3:, :, :]
        
        # Split image
        _, _, h, w = face_sequences_3ch.shape
        split_idx = h // 2
        
        # Prepare bottom input
        bottom_half = face_sequences_3ch[:, :, split_idx:, :]
        bottom_ref_channels = face_sequences_ref[:, :, split_idx:, :]
        bottom_input = torch.cat([bottom_half, bottom_ref_channels], dim=1)
        
        # First stage: Main network
        # Encode features
        audio_features = self.audio_encoder(audio_sequences)
        bottom_features = self.bottom_encoder(bottom_input)
        
        # Fuse modalities in bottleneck
        fused_features = self.bottleneck(bottom_features, audio_features)
        
        # Decode to generate bottom half
        bottom_output = self.bottom_decoder(fused_features)
        generated_bottom = self.output_conv(bottom_output)
        
        # Second stage: Enhancer network (if enabled)
        if train_enhancer:
            # Create full image from original top half and generated bottom half
            first_stage_output = torch.cat([original_top_half, generated_bottom], dim=2)
            
            # Enhancer input: first stage output + reference channels
            # This creates the input as: [first_stage_output, face_sequences_ref]
            enhancer_combined_input = torch.cat([first_stage_output, face_sequences_ref], dim=1)
            
            # Process with enhancer components
            enhancer_audio_features = self.enhancer_audio_encoder(audio_sequences)
            enhancer_features = self.enhancer_encoder(enhancer_combined_input)
            
            # Fuse modalities in enhancer bottleneck
            enhancer_fused_features = self.enhancer_bottleneck(enhancer_features, enhancer_audio_features)
            
            # Decode with enhancer
            enhancer_output = self.enhancer_decoder(enhancer_fused_features)
            enhanced_bottom = self.enhancer_output_conv(enhancer_output)
            
            # Extract bottom half from enhanced output
            enhanced_bottom = enhanced_bottom[:, :, split_idx:, :]
            
            # Use enhanced output
            final_bottom = enhanced_bottom
        else:
            # Use main network output directly
            final_bottom = generated_bottom
        
        # Handle output formatting
        if input_dim_size > 4:
            bottom_outputs = torch.split(final_bottom, B, dim=0)
            bottom_outputs = torch.stack(bottom_outputs, dim=2)
            original_top_half = torch.split(original_top_half, B, dim=0)
            original_top_half = torch.stack(original_top_half, dim=2)
        else:
            bottom_outputs = final_bottom
            
        
        if training:
            return None, bottom_outputs, None
        else:
            full_image = torch.cat([original_top_half, bottom_outputs], dim=2)
            return full_image, None, None
    
    def diffuse(self, x, channels_to_mask=3):
        """Diffusion with ellipse mask"""
        b, c, h, w = x.shape
        split_idx = h // 2
        top_half = x[:, :, :split_idx, :]
        bottom_half = x[:, :, split_idx:, :]
        
        rgb_channels = bottom_half[:, :channels_to_mask, :, :]
        other_channels = bottom_half[:, channels_to_mask:, :, :]
        
        mask = self.generate_ellipse_mask(h, w, split_idx, x.device)
        mask = mask.unsqueeze(0).unsqueeze(0).repeat(b, channels_to_mask, 1, 1)
        
        masked_rgb = rgb_channels * (1 - mask)
        noisy_bottom = torch.cat([masked_rgb, other_channels], dim=1)
        return torch.cat([top_half, noisy_bottom], dim=2)
    
    def generate_ellipse_mask(self, h, w, split_idx, device):
        """Generate ellipse mask for bottom half"""
        y_bottom = torch.linspace(0, 1, h - split_idx, device=device) * 2 - 1
        x_coord = torch.linspace(-1, 1, w, device=device)
        y_grid, x_grid = torch.meshgrid(y_bottom, x_coord, indexing='ij')
        
        dx, dy = self.ellipse_params['center']
        a_ratio, b_ratio = self.ellipse_params['axes']
        adjusted_a = a_ratio * (w / h)
        
        ellipse_mask = ((x_grid - dx)/adjusted_a)**2 + ((y_grid - dy)/b_ratio)**2 <= 1.0
        return ellipse_mask.float()

# === Improved Encoder Components ===

class ResidualBlock(nn.Module):
    """Improved residual block with better gradient flow"""
    def __init__(self, channels, kernel_size=3, dropout=0.1):
        super().__init__()
        padding = kernel_size // 2
        
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.GELU()
        
    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual  # True residual connection
        return self.activation(out)

class DownsampleBlock(nn.Module):
    """Downsampling block with proper gradient flow"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2):
        super().__init__()
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                             stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.GELU()
        
        # Skip connection for channel mismatch
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip = nn.Identity()
    
    def forward(self, x):
        residual = self.skip(x)
        out = self.conv(x)
        out = self.bn(out)
        out += residual
        return self.activation(out)

class BottomEncoder(nn.Module):
    """Improved bottom half encoder"""
    def __init__(self):
        super().__init__()
        
        # Initial convolution
        self.init_conv = nn.Sequential(
            nn.Conv2d(6, 64, 7, padding=3),  # 12 input channels
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        
        # Downsampling path
        self.down1 = DownsampleBlock(64, 96, stride=2)   # 192x192 -> 96x96
        self.res1 = ResidualBlock(96)
        self.initial_to_down2 = nn.Sequential(
          nn.Conv2d(64, 96, 1, stride=2),  # Skip connection
          nn.BatchNorm2d(96),
          nn.GELU()
        )
        
        self.down2 = DownsampleBlock(96, 128, stride=2)  # 96x96 -> 48x48
        self.res2 = ResidualBlock(128)
        self.down2_to_down3 = nn.Sequential(
          nn.Conv2d(96, 128, 1, stride=2),  # Skip connection
          nn.BatchNorm2d(128),
          nn.GELU()
        )
        
        self.down3 = DownsampleBlock(128, 192, stride=2) # 48x48 -> 24x24
        self.res3 = ResidualBlock(192)
        
        self.down4 = DownsampleBlock(192, 256, stride=2) # 24x24 -> 12x12
        self.res4 = ResidualBlock(256)
        
        # Feature maps for skip connections
        self.feature_maps = {}
    
    def forward(self, x):
        features = []
        
        x = self.init_conv(x)
        features.append(x)  # 192x384
        initial_skip = self.initial_to_down2(x)
        
        x = self.down1(x)  # 96x192
        x = self.res1(x)
        features.append(x)
        
        down2_skip = self.down2_to_down3(x + initial_skip)
        x = self.down2(x + initial_skip)  # 48x96
        x = self.res2(x)
        features.append(x)
        
        
        x = self.down3(x + down2_skip)  # 24x48
        x = self.res3(x)
        features.append(x)
        
        x = self.down4(x)  # 12x24
        x = self.res4(x)
        features.append(x)
        
        return features

class AudioEncoder(nn.Module):
    """Improved audio encoder"""
    def __init__(self):
        super().__init__()
        
        self.encoder = nn.Sequential(
            # Initial layers
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            
            # Downsample with residual
            DownsampleBlock(32, 64, stride=(2, 1)),  # Temporal downsampling
            ResidualBlock(64),
            
            DownsampleBlock(64, 128, stride=1),
            ResidualBlock(128),
            
            DownsampleBlock(128, 192, stride=2),
            ResidualBlock(192),
            
            # Final projection
            nn.AdaptiveAvgPool2d((12, 24)),  # Match visual features
            nn.Conv2d(192, 256, 1),
            nn.BatchNorm2d(256),
            nn.GELU()
        )
    
    def forward(self, x):
        return self.encoder(x)

# === Improved Bottleneck ===

class CrossModalAttention(nn.Module):
    """Simple cross-modal attention"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        self.visual_query = nn.Conv2d(channels, channels, 1)
        self.audio_key = nn.Conv2d(channels, channels, 1)
        self.audio_value = nn.Conv2d(channels, channels, 1)
        
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.ones(1) * 0.5)
    
    def forward(self, visual_feat, audio_feat):
        batch_size, channels, height, width = visual_feat.shape
        
        # Resize audio to match visual if needed
        if audio_feat.shape[-2:] != visual_feat.shape[-2:]:
            audio_feat = F.interpolate(audio_feat, size=(height, width), mode='bilinear')
        
        # Compute attention
        query = self.visual_query(visual_feat).view(batch_size, channels, -1)
        key = self.audio_key(audio_feat).view(batch_size, channels, -1)
        value = self.audio_value(audio_feat).view(batch_size, channels, -1)
        
        attention = torch.bmm(query.transpose(1, 2), key)  # [B, HW, HW]
        attention = self.softmax(attention)
        
        attended_audio = torch.bmm(value, attention.transpose(1, 2))
        attended_audio = attended_audio.view(batch_size, channels, height, width)
        
        # Fuse with visual features
        #print('The gamma', self.gamma)  # Debugging line
        positive_gamma = F.relu(self.gamma) + 1e-6
        return visual_feat + positive_gamma * attended_audio

class CrossModalBottleneck(nn.Module):
    """Proper multi-level cross-modal fusion"""
    def __init__(self):
        super().__init__()
        
        # Attention at multiple levels
        self.attention1 = CrossModalAttention(256)  # For bottleneck
        self.attention2 = CrossModalAttention(192)  # For mid-level features
        self.attention3 = CrossModalAttention(128)  # For higher-level features
        
        # Audio projection for different levels
        self.audio_proj_mid = nn.Sequential(
            nn.Conv2d(256, 192, 1),
            nn.BatchNorm2d(192),
            nn.GELU()
        )
        
        self.audio_proj_high = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.GELU()
        )
        
        # Bottleneck processing
        self.bottleneck_conv = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256)
        )
    
    def forward(self, visual_features, audio_features):
        """
        visual_features: [x0, x1, x2, x3, bottleneck]
        - x0: [B, 64, 192, 384]   (shallow)
        - x1: [B, 96, 96, 192]    
        - x2: [B, 128, 48, 96]    
        - x3: [B, 192, 24, 48]    
        - bottleneck: [B, 256, 12, 24]  (deepest)
        """
        x0, x1, x2, x3, bottleneck = visual_features
        
        # Process bottleneck level (deepest features)
        bottleneck = self.attention1(bottleneck, audio_features)
        bottleneck = self.bottleneck_conv(bottleneck)
        
        # Process mid-level features (x3)
        audio_mid = F.interpolate(audio_features, size=x3.shape[-2:], mode='bilinear')
        audio_mid = self.audio_proj_mid(audio_mid)
        x3 = self.attention2(x3, audio_mid)
        
        # Process higher-level features (x2)
        audio_high = F.interpolate(audio_features, size=x2.shape[-2:], mode='bilinear')
        audio_high = self.audio_proj_high(audio_high)
        x2 = self.attention3(x2, audio_high)
        
        # Return all updated features
        return [x0, x1, x2, x3, bottleneck]

class UpsampleBlock(nn.Module):
    """Upsampling block with proper gradient flow"""
    def __init__(self, in_channels, out_channels, kernel_size=3, scale_factor=2):
        super().__init__()
        
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.GELU()
        
        # Skip connection processing
        self.skip_conv = nn.Conv2d(out_channels, out_channels, 1)
        
    def forward(self, x, skip=None):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.bn(x)
        
        if skip is not None:
            # Ensure skip connection matches dimensions
            if skip.shape[-2:] != x.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear')
            x = x + self.skip_conv(skip)
        
        return self.activation(x)

class BottomDecoder(nn.Module):
    """Improved bottom half decoder"""
    def __init__(self):
        super().__init__()
        
        # Upsampling path
        self.up4 = UpsampleBlock(256, 192)  # 12x24 -> 24x48
        self.res4 = ResidualBlock(192)
        
        self.up3 = UpsampleBlock(192, 128)  # 24x48 -> 48x96
        self.res3 = ResidualBlock(128)
        
        self.up2 = UpsampleBlock(128, 96)   # 48x96 -> 96x192
        self.res2 = ResidualBlock(96)
        self.up2_vertical_cat = nn.Sequential(
            nn.Conv2d(96 + 96, 192, 3, 1, 1),
            nn.BatchNorm2d(192),
            nn.GELU(),
            nn.Conv2d(96 + 96, 96, 3, 1, 1),
            nn.BatchNorm2d(96),
            nn.GELU()
        )
        
        self.up1 = UpsampleBlock(96, 64)    # 96x192 -> 192x384
        self.res1 = ResidualBlock(64)
        self.up1_vertical_cat = nn.Sequential(
            nn.Conv2d(64 + 64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.GELU()
        )
    
    def forward(self, features):
        # features: list from encoder [x0, x1, x2, x3, bottleneck]
        x0, x1, x2, x3, bottleneck = features
        
        # Decode with skip connections
        x = self.up4(bottleneck, x3)  # Use x3 as skip
        x = self.res4(x)
        
        x = self.up3(x, x2)  # Use x2 as skip
        x = self.res3(x)
        
        x = self.up2(x, x1)  # Use x1 as skip        
        concated = torch.cat([x, x1], dim=1)
        concated = self.up2_vertical_cat(concated)
        x = self.res2(concated)
        
        x = self.up1(x, x0)  # Use x0 as skip
        concated = torch.cat([x, x0], dim=1)
        concated = self.up1_vertical_cat(concated)
        x = self.res1(concated)
        
        return x

# === Utility Functions ===

def init_weights(m):
    """Proper weight initialization"""
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

# Initialize the model
def create_model(print_gradients=False):
    model = ResUNet384V8(print_gradients=print_gradients)
    model.apply(init_weights)
    return model
