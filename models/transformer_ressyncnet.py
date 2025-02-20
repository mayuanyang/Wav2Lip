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
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

class TransformerResSyncnet(nn.Module):
    def __init__(self, num_encoder_layers=6, embed_dim=256, num_heads=8, dropout=0.1):
        super(TransformerResSyncnet, self).__init__()
        
        # Face Encoder
        self.face_encoder = models.resnet50(pretrained=True)
        self.face_encoder.conv1 = modify_resnet_conv1(self.face_encoder, in_channels=15)
        self.face_encoder.fc = nn.Identity()
        
        # Audio Encoder
        self.audio_encoder = models.resnet50(pretrained=True)
        self.audio_encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.audio_encoder.fc = nn.Identity()
        
        # Projection Layers (to match embed_dim)
        self.face_proj = nn.Linear(2048, embed_dim)
        self.audio_proj = nn.Linear(2048, embed_dim)
        
        # Initialize projections
        self.face_proj.apply(initialize_weights)
        self.audio_proj.apply(initialize_weights)
        
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=num_heads, dropout=0.1),
            num_layers=num_encoder_layers
        )
        
        # Fully Connected Layers for Classification
        
        self.relu = nn.LeakyReLU(0.01, inplace=True)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, 2)
        
        # Initialize FC layers
        self.fc2.apply(initialize_weights)
        
    def forward(self, face, audio):
        # Encode face and audio
        face_embedding = self.face_encoder(face)    # (batch_size, 2048)
        audio_embedding = self.audio_encoder(audio) # (batch_size, 512)
        
        # Project embeddings to common dimension
        face_proj = self.face_proj(face_embedding)   # (batch_size, embed_dim)
        audio_proj = self.audio_proj(audio_embedding) # (batch_size, embed_dim)
        
        # Normalize embeddings
        face_proj = F.normalize(face_proj, p=2, dim=1)
        audio_proj = F.normalize(audio_proj, p=2, dim=1)
        

        # Concatenate lip frames and audio features
        combined = torch.cat((face_proj, audio_proj), dim=1)
        
        # Make sure combined is 1-dimensional
        combined = combined.view(combined.size(0), -1)

        # Pass through the Transformer encoder, the input size is 1024
        transformer_output = self.transformer_encoder(combined)
        out = self.relu(transformer_output)
        out = self.fc2(out)
        out = self.dropout1(out)
        
        return out, audio_embedding, face_embedding
