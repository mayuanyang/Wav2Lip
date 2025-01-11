import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models as models

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
    def __init__(self, num_heads, num_encoder_layers):
        super(TransformerEfficientNetB3Syncnet, self).__init__()
        
        # Load the pretrained EfficientNet-B3 model for face encoding
        self.face_encoder = models.efficientnet_b3(pretrained=True)
        
        # Modify the first convolutional layer to accept 15 input channels
        self.face_encoder.features[0][0] = modify_efficientnet_conv1(self.face_encoder, in_channels=15)
        
        # Remove the classifier to get embeddings instead of class scores
        self.face_encoder.classifier = nn.Identity()
        
        # Similarly, load the pretrained EfficientNet-B3 model for audio encoding
        self.audio_encoder = models.efficientnet_b3(pretrained=True)
        
        # Modify the first convolutional layer to accept 1 input channel
        self.audio_encoder.features[0][0] = modify_efficientnet_conv1(self.audio_encoder, in_channels=1)
        
        # Initialize audio_encoder's conv1 weights (since it's 1 channel)
        with torch.no_grad():
            if 1 < 3:
                # If in_channels < 3, copy the first `in_channels` weights
                self.audio_encoder.features[0][0].weight.copy_(
                    self.audio_encoder.features[0][0].weight[:, :1, :, :]
                )
            elif 1 == 3:
                # If in_channels == 3, weights are already copied in modify_efficientnet_conv1
                pass
            else:
                # If in_channels > 3, already initialized in modify_efficientnet_conv1
                pass
        
        # Remove the classifier to get embeddings instead of class scores
        self.audio_encoder.classifier = nn.Identity()
        
        # Define the fully connected layers based on EfficientNet-B3's output features (1536)
        self.fc1 = nn.Linear(3072, 768)
        self.fc3 = nn.Linear(768, 2)
        
        # Initialize weights
        self.fc1.apply(initialize_weights)
        self.fc3.apply(initialize_weights)

        # Transformer Encoder
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=768, nhead=num_heads, dropout=0.3),
            num_layers=num_encoder_layers
        )
        
        self.relu = nn.LeakyReLU(0.01, inplace=True)
        
    def forward(self, face, audio):
        # Encode face input
        face_embedding = self.face_encoder(face)  # Shape: (batch_size, 1536)
        
        # Encode audio input
        audio_embedding = self.audio_encoder(audio)  # Shape: (batch_size, 1536)
        
        # Normalize embeddings
        face_embedding = F.normalize(face_embedding, p=2, dim=1)
        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        
        # Concatenate embeddings
        combined = torch.cat((face_embedding, audio_embedding), dim=1)  # Shape: (batch_size, 768)

        combined = self.fc1(combined)
        
        # Reshape for Transformer: (sequence_length, batch_size, embedding_dim)
        combined = combined.unsqueeze(0)  # Shape: (1, batch_size, 768)
        
        transformer_output = self.transformer_encoder(combined)  # Shape: (1, batch_size, 768)
        out = self.relu(transformer_output)
        out = self.fc3(out.squeeze(0))  # Shape: (batch_size, 2)
        
        return out, audio_embedding, face_embedding