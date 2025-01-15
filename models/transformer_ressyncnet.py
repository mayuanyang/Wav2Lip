import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models as models
from .cross_attention import CrossAttention

def modify_resnet_conv1(resnet, in_channels):
    """
    Modify the first convolutional layer of ResNet to accept `in_channels` input channels.

    Args:
        resnet (nn.Module): The ResNet model to modify.
        in_channels (int): The number of input channels.

    Returns:
        nn.Module: The modified ResNet model.
    """
    old_conv = resnet.conv1
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
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d) :
        nn.init.kaiming_normal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

class TransformerResSyncnet(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, dropout=0.1):
        super(TransformerResSyncnet, self).__init__()
        
        # Face Encoder
        self.face_encoder = models.resnet50(pretrained=True)
        self.face_encoder.conv1 = modify_resnet_conv1(self.face_encoder, in_channels=15)
        self.face_encoder.fc = nn.Identity()
        
        # Audio Encoder
                # Similarly, modify the audio_encoder if necessary
        self.audio_encoder = models.resnet18(pretrained=True)
        self.audio_encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.audio_encoder.fc = nn.Identity()
        
        # Projection Layers (optional, to match embed_dim)
        self.face_proj = nn.Linear(2048, embed_dim)
        self.audio_proj = nn.Linear(512, embed_dim)
        
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
        
        # normalise them
        audio_proj = F.normalize(audio_proj, p=2, dim=1)
        face_proj = F.normalize(face_proj, p=2, dim=1)
        
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
