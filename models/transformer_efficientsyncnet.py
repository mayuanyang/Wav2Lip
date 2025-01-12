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
    def __init__(self, embed_dim=1536, num_heads=8, dropout=0.1):
        super(TransformerEfficientNetB3Syncnet, self).__init__()
        
        # Face Encoder
        self.face_encoder = models.efficientnet_b3(pretrained=True)
        self.face_encoder.features[0][0] = modify_efficientnet_conv1(self.face_encoder, in_channels=15)
        self.face_encoder.classifier = nn.Identity()
        
        # Audio Encoder
        self.audio_encoder = models.efficientnet_b3(pretrained=True)
        self.audio_encoder.features[0][0] = modify_efficientnet_conv1(self.audio_encoder, in_channels=1)
        self.audio_encoder.classifier = nn.Identity()
        
        # Projection Layers (optional, to match embed_dim)
        self.face_proj = nn.Linear(1536, embed_dim)
        self.audio_proj = nn.Linear(1536, embed_dim)
        
        # Initialize projections
        self.face_proj.apply(initialize_weights)
        self.audio_proj.apply(initialize_weights)
        
        # Cross-Attention Layer
        self.cross_attn = CrossAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        
        # Fully Connected Layers for Classification
        self.fc1 = nn.Linear(embed_dim, 512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, 2)
        self.dropout2 = nn.Dropout(dropout)
        
        # Initialize FC layers
        self.fc1.apply(initialize_weights)
        self.fc2.apply(initialize_weights)
        
    def forward(self, face, audio):
        # Encode face and audio
        face_embedding = self.face_encoder(face)    # (batch_size, 1536)
        audio_embedding = self.audio_encoder(audio) # (batch_size, 1536)
        
        # Project embeddings to common dimension
        face_proj = self.face_proj(face_embedding)   # (batch_size, embed_dim)
        audio_proj = self.audio_proj(audio_embedding) # (batch_size, embed_dim)
        
        # Apply Cross-Attention
        # Let's attend face to audio and audio to face, then combine
        face_to_audio = self.cross_attn(face_proj, audio_proj, audio_proj)  # (batch_size, embed_dim)
        audio_to_face = self.cross_attn(audio_proj, face_proj, face_proj)  # (batch_size, embed_dim)
        
        # Combine the attended embeddings
        combined = face_to_audio + audio_to_face  # (batch_size, embed_dim)
        
        # Classification Layers
        combined = self.fc1(combined)             # (batch_size, 512)
        combined = self.relu1(combined)
        combined = self.dropout1(combined)
        out = self.fc2(combined)                   # (batch_size, 2)
        out = self.dropout2(out)
        
        return out, audio_embedding, face_embedding