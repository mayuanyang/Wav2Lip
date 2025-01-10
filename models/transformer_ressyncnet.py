import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models as models

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

class TransformerResSyncnet(nn.Module):
    def __init__(self, num_heads, num_encoder_layers):
        super(TransformerResSyncnet, self).__init__()
        
        # Load the pretrained ResNet18 model for face encoding
        self.face_encoder = models.resnet50(pretrained=True)
        
        # Modify the first convolutional layer to accept 15 input channels
        self.face_encoder.conv1 = modify_resnet_conv1(self.face_encoder, in_channels=15)
        
        # Remove the fully connected layer
        self.face_encoder.fc = nn.Identity()
        
        # Similarly, modify the audio_encoder if necessary
        self.audio_encoder = models.resnet18(pretrained=True)
        self.audio_encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Initialize audio_encoder's conv1 weights (since it's 1 channel)
        with torch.no_grad():
            self.audio_encoder.conv1.weight = nn.Parameter(
                self.audio_encoder.conv1.weight.sum(dim=1, keepdim=True)
            )
        
        self.audio_encoder.fc = nn.Identity()
        
        # Transformer Encoder
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=num_heads, dropout=0.1),
            num_layers=num_encoder_layers
        )
        
        self.relu = nn.LeakyReLU(0.01, inplace=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(2048, 256)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(512, 2)
        
    def forward(self, face, audio):
        face_embedding = self.face_encoder(face)  # Shape: (batch_size, 512)
        audio_embedding = self.audio_encoder(audio)  # Shape: (batch_size, 512)
        
        face_embedding = self.fc1(face_embedding)  # Shape: (batch_size, 256)
        audio_embedding = self.fc2(audio_embedding)  # Shape: (batch_size, 256)
        
        # Normalize embeddings
        face_embedding = F.normalize(face_embedding, p=2, dim=1)
        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        
        # Concatenate embeddings
        combined = torch.cat((face_embedding, audio_embedding), dim=1)  # Shape: (batch_size, 512)
        
        # Reshape for Transformer: (sequence_length, batch_size, embedding_dim)
        combined = combined.unsqueeze(0)  # Shape: (1, batch_size, 512)
        
        transformer_output = self.transformer_encoder(combined)  # Shape: (1, batch_size, 512)
        out = self.relu(transformer_output)
        out = self.fc3(out.squeeze(0))  # Shape: (batch_size, 2)
        
        return out, audio_embedding, face_embedding
