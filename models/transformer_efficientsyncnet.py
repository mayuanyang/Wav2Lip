import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models as models
from .cross_attention import CrossAttention

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

def initialize_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


    
class TransformerEfficientNetB3Syncnet(nn.Module):
    def __init__(self, num_cross_attn_layers=4, embed_dim=256, num_heads=8, dropout=0.1):
        super(TransformerEfficientNetB3Syncnet, self).__init__()
        
        # Face Encoder
        self.face_encoder = models.efficientnet_b3(pretrained=True)
        self.face_encoder.classifier = nn.Identity()
        
        # Audio Encoder
        self.audio_encoder = models.efficientnet_b3(pretrained=True)
        self.audio_encoder.features[0][0] = modify_efficientnet_conv1(self.audio_encoder, in_channels=1)
        self.audio_encoder.classifier = nn.Identity()
        
        # Projection Layers (optional, to match embed_dim)
        audio_scale_factor = 1000
        num_of_frames = 5
        audio_embed_dim = int(audio_scale_factor/num_of_frames)
        
        self.face_proj = nn.Linear(1536, embed_dim)
        self.audio_proj = nn.Linear(1536, audio_scale_factor) # Can be divide by 5
        
                
        # 新增：各自模态的 self-attention 层
        self.face_self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        
        self.audio_self_attn = nn.MultiheadAttention(embed_dim=audio_embed_dim, num_heads=num_heads)
        
        self.fuse_proj = nn.Linear(456, embed_dim*2)
        
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim * 2, num_heads, dropout=dropout) 
            for _ in range(num_cross_attn_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim*2) for _ in range(num_cross_attn_layers)
        ])
                
        
        # Final classification head.
        # We pool tokens for each modality separately, then concatenate their global features.
        self.classifier = nn.Sequential(
            # nn.Linear(1536, 512), for 192
            nn.Linear(2560, 256), 
            nn.LeakyReLU(0.2, inplace=False),
            nn.Dropout(p=0.1),
            nn.Linear(256, 1)  # binary classification output
        )
        
        
    def forward(self, face, audio):
        num_of_frames = 5
        
        save_every_s_steps = 1000
        
        batch_size = face.shape[0]
        
        # --- Process audio modality ---
        audio_embedding = self.audio_encoder(audio) # (batch_size, 1536)
                

        ### ---视觉分支--- ###        
        face_embedding = face.view(batch_size * num_of_frames ,3 ,192 ,384)
        
        # Encode face and audio
        face_embedding = self.face_encoder(face_embedding)    # (batch_size, 1536)
        
        # Project embeddings to common dimension
        face_proj = self.face_proj(face_embedding)   # (batch_size, embed_dim)
        audio_proj = self.audio_proj(audio_embedding) # (batch_size, embed_dim)
        
                
        face_seq = face_proj.view(batch_size, num_of_frames, -1).permute(1, 0, 2)
        audio_seq = audio_proj.view(batch_size, 5, -1).permute(1, 0, 2)
        
        face_self_out, _ = self.face_self_attn(
            query=face_seq,
            key=face_seq,
            value=face_seq
        )
        
        # Audio 的 self-attention
        audio_self_out, _ = self.audio_self_attn(
            query=audio_seq,
            key=audio_seq,
            value=audio_seq
        )

        # --- 合并特征 ---
        combined = torch.cat((face_self_out, audio_self_out), dim=2)  # 沿特征维度拼接
                
        combined = self.fuse_proj(combined)
        
        attn_output = combined
        
        for layer, layer_norm in zip(self.cross_attn_layers, self.layer_norms):
            # Apply multi-head attention
            attn_output, _ = layer(attn_output, combined, combined)
            # Apply LayerNorm after attention
            attn_output = layer_norm(attn_output + combined)
        
        attn_output = attn_output.permute(1, 0, 2).reshape(batch_size, -1)
        
        output = self.classifier(attn_output)
        
        return output, None, None