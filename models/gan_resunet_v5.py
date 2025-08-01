import torch
import torch.nn as nn
import torch.nn.functional as F
from .resunet_v5 import ResUNet384V5

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        # Audio encoder (simplified)
        self.audio_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((16, 16))  # Reduce to manageable size
        )
        
        # Image encoder (simplified)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # 192x192
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 96x96
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 48x48
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 24x24
            nn.LeakyReLU(0.2),
            
            nn.AdaptiveAvgPool2d((16, 16))  # Reduce to same size as audio
        )
        
        # Fusion layer to combine audio and image features
        # After pooling: audio [B, 64, 16, 16] -> [B, 64*16*16] = [B, 16384]
        # After pooling: image [B, 256, 16, 16] -> [B, 256*16*16] = [B, 65536]
        # Combined: [B, 16384 + 65536] = [B, 81920]
        self.fusion = nn.Sequential(
            nn.Linear(16384 + 65536, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, audio_sequences, face_sequences):
        # Check if face_sequences is 5D (video) or 4D (image)
        if len(face_sequences.size()) > 4:
            # Handle 5D inputs (batch, channels, time, height, width)
            B, C, T, H, W = face_sequences.size()
            
            # Flatten time dimension into batch dimension for face sequences
            face_sequences_flat = face_sequences.transpose(1, 2).contiguous().view(B * T, C, H, W)
            
            # Expand audio sequences to match the flattened face sequences
            # audio_sequences is [B, 1, 80, 16], we need [B*T, 1, 80, 16]
            audio_sequences_expanded = audio_sequences.unsqueeze(2).expand(-1, -1, T, -1, -1).contiguous().view(B * T, -1, audio_sequences.size(2), audio_sequences.size(3))
            
            # Encode audio
            audio_features = self.audio_encoder(audio_sequences_expanded)  # [B*T, 64, 16, 16]
            audio_features = audio_features.view(audio_features.size(0), -1)  # Flatten [B*T, 64*16*16] = [B*T, 16384]
            
            # Encode image
            image_features = self.image_encoder(face_sequences_flat)  # [B*T, 256, 16, 16]
            image_features = image_features.view(image_features.size(0), -1)  # Flatten [B*T, 256*16*16] = [B*T, 65536]
            
            # Combine features
            combined_features = torch.cat((audio_features, image_features), dim=1)
            
            # Final classification
            validity = self.fusion(combined_features)
            
            # Reshape output back to [B, T]
            validity = validity.view(B, T)
            
            # Take the average over time dimension
            validity = validity.mean(dim=1)
        else:
            # Handle 4D inputs (standard image inputs)
            # Encode audio
            audio_features = self.audio_encoder(audio_sequences)  # [B, 64, 16, 16]
            audio_features = audio_features.view(audio_features.size(0), -1)  # Flatten [B, 64*16*16] = [B, 16384]
            
            # Encode image
            image_features = self.image_encoder(face_sequences)  # [B, 256, 16, 16]
            image_features = image_features.view(image_features.size(0), -1)  # Flatten [B, 256*16*16] = [B, 65536]
            
            # Combine features
            combined_features = torch.cat((audio_features, image_features), dim=1)
            
            # Final classification
            validity = self.fusion(combined_features)
        
        return validity

class Wav2LipGAN(nn.Module):
    def __init__(self):
        super(Wav2LipGAN, self).__init__()
        
        # Initialize generator and discriminator
        self.generator = ResUNet384V5()
        self.discriminator = Discriminator()

    def forward(self, audio_sequences, face_sequences, step=None):
        # Check if inputs are 5D (video) or 4D (image)
        input_dim_size = len(face_sequences.size())
        B = audio_sequences.size(0)
        
        # Handle 5D inputs (batch, channels, time, height, width)
        if input_dim_size > 4:
            # Flatten time dimension into batch dimension
            audio_sequences_flat = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)
            face_sequences_flat = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)
            
            # Generator forward pass
            generated_face_flat, t0, _ = self.generator(audio_sequences_flat, face_sequences_flat, step)
            
            # Reshape output back to 5D
            generated_face = torch.stack(torch.split(generated_face_flat, B, dim=0), dim=2)
        else:
            # Generator forward pass for 4D inputs
            generated_face, t0, _ = self.generator(audio_sequences, face_sequences, step)
            
        return generated_face, t0, None

# For standalone use or testing
if __name__ == "__main__":
    # Example usage
    model = Wav2LipGAN()
    print("Generator:")
    print(model.generator)
    print("\nDiscriminator:")
    print(model.discriminator)
