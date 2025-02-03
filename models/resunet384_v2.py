import torch
from torch import nn
from torch.nn import functional as F


from .conv import Conv2dTranspose, Conv2d, nonorm_Conv2d

class ResUNet384V2(nn.Module):
    def __init__(self, num_of_blocks=2):
        super(ResUNet384V2, self).__init__()
        self.face_encoder = nn.Sequential(
            
            Conv2d(12, 64, kernel_size=3, stride=1, padding=1), #192x192, 1+(3−1)×1=3
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), #192x192, 3+(3−1)×1=5
            
            Conv2d(64, 64, kernel_size=3, stride=2, padding=1), #96x96, 7
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), # 9

            Conv2d(64, 128, kernel_size=3, stride=(1, 2), padding=1), #94x47, 13
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), #94x47, 15
            
            Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # 47x24, 19
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True), # 47x24, 21  

            Conv2d(256, 64, kernel_size=3, stride=2, padding=1), # 24x 12, 25
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), # 24x 12, 27
          
            Conv2d(64, 32, kernel_size=3, stride=2, padding=1), # 24x 12, 25
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True), # 24x 12, 25
            
            Conv2d(32, 16, kernel_size=3, stride=2, padding=1), # 24x 12, 25
            Conv2d(16, 16, kernel_size=3, stride=1, padding=1, residual=True), # 24x 12, 25

            )

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1), # 80x16, 1+(3−1)×1=3
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True), # 3+(3-1)x1=5
            
            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1), # 27x16, 11+(3-1)x3=17
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), #15+(3-1)x1=19
            
            Conv2d(64, 64, kernel_size=3, stride=2, padding=1), # 14x8, 25+(3-1)x2=29
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), #29+(3-1)x1=31
            
            Conv2d(64, 128, kernel_size=3, stride=(2,1), padding=1), #7x8, 35+(3-1)x2=41
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), #41+(3-1)x1=43
            

            Conv2d(128, 256, kernel_size=3, stride=3, padding=1), #3x3, #47+(3-1)x3=51
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True), #51+(3-1)x1=53

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0), #3x3, 53+(3-1)x1=55
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0, residual=True),) #55+(3-1)x1=57

        self.fc1 = nn.Linear(1152, 256) 
        self.fc2 = nn.Linear(512, 256) 

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8, dropout=0.1),
            num_layers=2
        )

        self.projection = nn.Conv2d(512, 8, kernel_size=1, bias=False)  # 1x1 convolution

        self.relu = nn.LeakyReLU(0.01, inplace=True)
        self.unet = UNet()

        

    def forward(self, audio_embedding, face_embedding):
        
        B = audio_embedding.size(0)
        T = audio_embedding.size(1)

        input_dim_size = len(face_embedding.size())
        if input_dim_size > 4:
            audio_embedding = torch.cat([audio_embedding[:, i] for i in range(audio_embedding.size(1))], dim=0)
            face_embedding = torch.cat([face_embedding[:, :, i] for i in range(face_embedding.size(2))], dim=0)

        original_face = face_embedding

        face_embedding = self.face_encoder(face_embedding)
        audio_embedding = self.audio_encoder(audio_embedding)

        audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)
        face_embedding = face_embedding.view(face_embedding.size(0), -1)

        
        face_embedding = self.fc1(face_embedding)
        audio_embedding = self.fc2(audio_embedding)

        # normalise them
        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        face_embedding = F.normalize(face_embedding, p=2, dim=1)

        # Concatenate lip frames and audio features
        combined = torch.cat((face_embedding, audio_embedding), dim=1)
        
        # Pass through the Transformer encoder, the input size is 512
        transformer_output = self.transformer_encoder(combined)
        out = self.relu(transformer_output)

        out = out.view(B*T, 512, 1, 1)  # Shape: [10, 512, 1, 1] 10 is the BatchSzie * TimeStep
        
        
        projected = self.projection(out)  # Shape: [10, C, 1, 1]

        # Upsample to [10, C, 384, 384] using interpolation
        upsampled = torch.nn.functional.interpolate(projected, size=(384, 384), mode='bilinear', align_corners=False)  # Shape: [10, C, 384, 384]

        # Concatenate along the channel dimension
        upsampled = torch.cat([original_face, upsampled], dim=1)  # Shape: [10, 12 + C, 384, 384]

        output = self.unet(upsampled)

        if input_dim_size > 4:
            output = torch.split(output, B, dim=0) # [(B, C, H, W)]
            output = torch.stack(output, dim=2) # (B, C, T, H, W)
        
        return output
    


class UNet(nn.Module):
    def __init__(self, in_channels=20, out_channels=3):
        super(UNet, self).__init__()

        # Encoder
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Decoder
        self.decoder4 = self.upconv_block(1024, 512)
        self.decoder3 = self.upconv_block(512, 256)
        self.decoder2 = self.upconv_block(256, 128)
        self.decoder1 = self.upconv_block(128, 64)

        # Final layer
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)  # [10, 64, 384, 384]
        p1 = self.pool(e1)  # [10, 64, 192, 192]

        e2 = self.encoder2(p1)  # [10, 128, 192, 192]
        p2 = self.pool(e2)  # [10, 128, 96, 96]

        e3 = self.encoder3(p2)  # [10, 256, 96, 96]
        p3 = self.pool(e3)  # [10, 256, 48, 48]

        e4 = self.encoder4(p3)  # [10, 512, 48, 48]
        p4 = self.pool(e4)  # [10, 512, 24, 24]

        # Bottleneck
        bottleneck = self.bottleneck(p4)  # [10, 1024, 24, 24]

        # Decoder with skip connections
        d4 = self.decoder4(bottleneck)  # [10, 512, 48, 48]
        d4 = torch.cat([d4, e4], dim=1)  # [10, 1024, 48, 48]
        d4 = self.conv_block(1024, 512)(d4)  # [10, 512, 48, 48]

        d3 = self.decoder3(d4)  # [10, 256, 96, 96]
        d3 = torch.cat([d3, e3], dim=1)  # [10, 512, 96, 96]
        d3 = self.conv_block(512, 256)(d3)  # [10, 256, 96, 96]

        d2 = self.decoder2(d3)  # [10, 128, 192, 192]
        d2 = torch.cat([d2, e2], dim=1)  # [10, 256, 192, 192]
        d2 = self.conv_block(256, 128)(d2)  # [10, 128, 192, 192]

        d1 = self.decoder1(d2)  # [10, 64, 384, 384]
        d1 = torch.cat([d1, e1], dim=1)  # [10, 128, 384, 384]
        d1 = self.conv_block(128, 64)(d1)  # [10, 64, 384, 384]

        # Final layer
        output = self.final_conv(d1)  # [10, 3, 384, 384]
        return output
