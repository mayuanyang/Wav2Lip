import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models as models
from .cross_attention import CrossAttention
from transformers import ASTModel, ASTConfig, ASTFeatureExtractor

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
    def __init__(
        self,
        num_encoder_layers=6,
        embed_dim=512,  # Updated to 512 to match Transformer d_model
        num_heads=8,
        dropout=0.1,
        num_classes=2   # Adjust as per your classification task
    ):
        super(TransformerResSyncnet, self).__init__()
        
        self.embed_dim = embed_dim  # For clarity

        # --- Face Encoder ---
        self.face_encoder = models.resnet50(pretrained=True)
        self.face_encoder.conv1 = modify_resnet_conv1(self.face_encoder, in_channels=3)
        self.face_encoder.fc = nn.Identity()  # Remove the final classification layer

        # --- Audio Encoder ---
        ast_config = ASTConfig()
        self.feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.audio_model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.audio_model.eval()
        self.audio_proj = nn.Linear(ast_config.hidden_size, embed_dim)
        #self.audio_encoder.eval()  # Freeze if not fine-tuning

        
        # --- Projection Layers (to match embed_dim) ---
        self.face_proj = nn.Linear(2048, embed_dim)
        #self.audio_proj = nn.Linear(2048, embed_dim)
        
        # Initialize projections
        self.face_proj.apply(initialize_weights)
        self.audio_proj.apply(initialize_weights)
        
        # --- Learnable CLS Token ---
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # (1, 1, 512)
        nn.init.normal_(self.cls_token, std=0.02)  # Initialize with normal distribution

        self.positional_encoding = nn.Parameter(torch.zeros(1, 1220, embed_dim))
        
        # --- Transformer Encoder ---
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout),
            num_layers=num_encoder_layers
        )
        
        # --- Fully Connected Layers for Classification ---
        self.relu = nn.LeakyReLU(0.01, inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.fc_classification = nn.Linear(embed_dim, num_classes)
        
        # Initialize FC layers
        self.fc_classification.apply(initialize_weights)
        
    def forward(self, face, audio):
        """
        Args:
            face (torch.Tensor): Tensor of shape (batch_size, 15, H, W)
            audio (torch.Tensor): Tensor of shape (batch_size, 1, H_a, W_a)

        Returns:
            torch.Tensor: Classification logits of shape (batch_size, num_classes)
            torch.Tensor: Audio embedding of shape (batch_size, 2048)
            torch.Tensor: Face embedding of shape (batch_size, 2048)
        """
        B = face.size(0)  # Batch size
        
        input_dim_size = len(face.size())
        if input_dim_size > 4:
            audio = torch.cat([audio[:, i] for i in range(audio.size(1))], dim=0)
            
        
        # --- Split Face Tensor into 5 Images ---
        # Reshape from (B, 15, H, W) to (B, 5, 3, H, W)
        num_frames = 5
        channels_per_frame = 3
        face_frames = face.view(B, num_frames, channels_per_frame, face.size(2), face.size(3))
        
        # Merge batch and frame dimensions to process all frames in parallel: (B*5, 3, H, W)
        face_frames = face_frames.view(B * num_frames, channels_per_frame, face.size(2), face.size(3))
        
        # --- Encode Each Face Frame ---
        face_embeddings = self.face_encoder(face_frames)  # (B*5, 2048)
        
        # Reshape back to (B, 5, 2048)
        face_embeddings = face_embeddings.view(B, num_frames, -1)  # (B, 5, 2048)
        
        # Project face embeddings to embed_dim
        face_tokens = self.face_proj(face_embeddings)  # (B, 5, 512)
        
        # --- Encode Audio ---
                
        resized_audio = F.interpolate(audio, size=(128, 1024), mode='bilinear', align_corners=False).squeeze(1)
        
        #input_audio_3ch = audio.repeat(1, 3, 1, 1)  # New shape: [29, 3, 80, 16]

        # Optionally, preprocess the input (if needed) using the feature extractor.
        # Here, we assume that your input is already a proper spectrogram.
        # If needed, you can convert it to numpy and back:
        # inputs = feature_extractor(input_audio_3ch.numpy(), return_tensors="pt")

        # Forward pass through the model
        audio_outputs = self.audio_model(resized_audio)

        # The model returns a ModelOutput. For ASTModel, `last_hidden_state` contains the embeddings.
        embeddings = audio_outputs.last_hidden_state  # Shape: [29, sequence_length, hidden_dim]
        
        # Project audio embedding to embed_dim
        audio_token = self.audio_proj(embeddings)  # (B, 512)
        
        # --- Prepare CLS Token ---
        # Expand cls_token to (B, 1, 512)
        cls_tokens = self.cls_token.expand(B, -1, -1)   # (B, 1, 512)
        
        # --- Concatenate Tokens ---
        # Combined tokens: [CLS] + 5 Face Tokens + Audio Token => (B, 7, 512)
        combined_tokens = torch.cat([cls_tokens, face_tokens, audio_token], dim=1)  # (B, 7, 512)
        
        # --- Transformer expects input shape (seq_len, batch_size, d_model) ---
        combined_tokens = combined_tokens.transpose(0, 1)  # (7, B, 512)
        
        positional_encoding = self.positional_encoding.transpose(0, 1).repeat(1, B, 1)  # (7, B, 512)
        combined_tokens = combined_tokens + positional_encoding
        
        # --- Transformer Encoding ---
        transformer_output = self.transformer_encoder(combined_tokens)  # (7, B, 512)

        aggregated_output = transformer_output.mean(dim=0)  # (B, 512)
        
        # --- Classification ---
        out = self.relu(aggregated_output)
        out = self.dropout(out)
        logits = self.fc_classification(out)  # (B, num_classes)
        
        return logits, None, None
