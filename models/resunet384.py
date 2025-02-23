import torch
from torch import nn
from torch.nn import functional as F


from .conv import Conv2dTranspose, Conv2d, nonorm_Conv2d

class ResUNet384(nn.Module):
    def __init__(self, num_of_blocks=2):
        super(ResUNet384, self).__init__()
        self.blocks = nn.ModuleList()

        for i in range(num_of_blocks):
            self.blocks.append(ProcessBlock384(12))
        
        print('The length of blocks', len(self.blocks))

        self.bn12 = nn.BatchNorm2d(12)

        self.bn3 = nn.BatchNorm2d(3)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
        self.output_block = ProcessBlock384(3)
                


    def forward(self, audio_sequences, face_sequences):
        temp_output = None
        face_input = face_sequences
        activation = "RELU"

        for i, block in enumerate(self.blocks):
            if temp_output is not None:
                face_input = face_sequences + temp_output
            if len(self.blocks) > 1:
                activation = "RELU"
            temp_output = self.forward_impl(audio_sequences, face_input, block.face_encoder_blocks, block.audio_encoder, block.face_decoder_blocks, block.output_block, 12, activation)
        
        activation = "NONE"
        if temp_output is not None:
          step2_face_sequences = face_sequences + temp_output
        else:
          step2_face_sequences = face_sequences
        outputs = self.forward_impl(audio_sequences, step2_face_sequences, self.output_block.face_encoder_blocks, self.output_block.audio_encoder, self.output_block.face_decoder_blocks, self.output_block.output_block, 3, activation)

        outputs = torch.sigmoid(outputs)
      
        return outputs

    def forward_impl(self, audio_sequences, face_sequences, face_encoder_blocks, audio_encoder, face_decoder_blocks, output_block, output_channels, activation):
        # audio_sequences = (B, T, 1, 80, 16)
        B = audio_sequences.size(0)

        input_dim_size = len(face_sequences.size())
        if input_dim_size > 4:
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)

        audio_embedding = audio_encoder(audio_sequences) # B, 512, 1, 1

        face_features = []
        this_face_sequence = face_sequences
        for f in face_encoder_blocks:
            this_face_sequence = f(this_face_sequence)
            face_features.append(this_face_sequence)

        
        x = audio_embedding
        # Apply attention before concatenation
        for i, f in enumerate(face_decoder_blocks):
            x = f(x)
            if face_features:
                skip = face_features.pop()
                x = torch.cat((x, skip), dim=1)

        x = output_block(x)
        if output_channels == 12:
          if activation == "RELU":
            x = self.leaky_relu(self.bn12(x))
          elif activation == "SIGMOID":
            x = torch.sigmoid(self.bn12(x))

        if input_dim_size > 4:
            x = torch.split(x, B, dim=0) # [(B, C, H, W)]
            outputs = torch.stack(x, dim=2) # (B, C, T, H, W)
        else:
            outputs = x
            
        return outputs


class ProcessBlock384(nn.Module):
    def __init__(self, output_block_channels) -> None:
        super(ProcessBlock384, self).__init__()
        '''
        Outpu = Input + (k-1) x S

        Where:
        Input is the receptive field size from the previous layer.
        k is the kernel size.
        S is the stride.
        '''

        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(12, 64, kernel_size=7, stride=1, padding=3), #1+(7−1)×1=7
                          Conv2d(64, 64, kernel_size=7, stride=1, padding=3, residual=True), #7+(7-1)*1=13
                          Conv2d(64, 64, kernel_size=7, stride=1, padding=3, residual=True), #13+(7-1)*1=19
                          ), # 384,384
            
            nn.Sequential(Conv2d(64, 192, kernel_size=3, stride=2, padding=1), #13+(7−1)×2=25
                          Conv2d(192, 192, kernel_size=3, stride=1, padding=1, residual=True), #25+(7-1)*1=31
                          Conv2d(192, 192, kernel_size=3, stride=1, padding=1, residual=True), #25+(7-1)*1=31
                          ), # 192,192

            nn.Sequential(Conv2d(192, 192, kernel_size=3, stride=2, padding=1), #31+(7−1)×2=43
              Conv2d(192, 192, kernel_size=3, stride=1, padding=1, residual=True), #49
              Conv2d(192, 192, kernel_size=3, stride=1, padding=1, residual=True), #49
              ), # 96,96

            nn.Sequential(Conv2d(192, 192, kernel_size=3, stride=2, padding=1), # 48,48, 49+(7−1)×2=61
            Conv2d(192, 192, kernel_size=3, stride=1, padding=1, residual=True), #67
            ),

            nn.Sequential(Conv2d(192, 192, kernel_size=3, stride=2, padding=1), # 24,24, 67+(7−1)×2=79
            Conv2d(192, 192, kernel_size=3, stride=1, padding=1, residual=True), #85
            ),

            nn.Sequential(Conv2d(192, 192, kernel_size=3, stride=2, padding=1), # 12,12, 85+(7−1)×2=97
            Conv2d(192, 192, kernel_size=3, stride=1, padding=1, residual=True), # 103
            ), 

            nn.Sequential(Conv2d(192, 192, kernel_size=5, stride=2, padding=2), # 6,6, 103+(5−1)×2=111
            Conv2d(192, 192, kernel_size=5, stride=1, padding=2, residual=True) # 115
            ), 

            nn.Sequential(Conv2d(192, 384, kernel_size=5, stride=2, padding=2), # 3,3, 115+(5−1)×2=123
            Conv2d(384, 384, kernel_size=5, stride=1, padding=2, residual=True), # 127
            ), 
            
            ]) # 45

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0, residual=True),)

        self.face_decoder_blocks = nn.ModuleList([
            

            nn.Sequential(Conv2dTranspose(512, 256, kernel_size=3, stride=1, padding=0), # 3,3
            Conv2d(256, 256, kernel_size=5, stride=1, padding=2, residual=True),),

            nn.Sequential(Conv2dTranspose(640, 320, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(320, 320, kernel_size=5, stride=1, padding=2, residual=True),), # 6, 6

            nn.Sequential(Conv2dTranspose(512, 352, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(352, 352, kernel_size=5, stride=1, padding=2, residual=True),
            ), # 12, 12

            nn.Sequential(Conv2dTranspose(544, 368, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(368, 368, kernel_size=5, stride=1, padding=2, residual=True),
            ), # 24, 24

            nn.Sequential(Conv2dTranspose(560, 376, kernel_size=3, stride=2, padding=1, output_padding=1), 
            Conv2d(376, 376, kernel_size=5, stride=1, padding=2, residual=True),
            ), # 48, 48

            nn.Sequential(Conv2dTranspose(568, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(128, 128, kernel_size=5, stride=1, padding=2, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            ), # 96,96
            
            nn.Sequential(
                Conv2dTranspose(320, 160, kernel_size=3, stride=2, padding=1, output_padding=1),
                Conv2d(160, 160, kernel_size=5, stride=1, padding=2, residual=True),
                Conv2d(160, 160, kernel_size=3, stride=1, padding=1, residual=True),
            ),
            nn.Sequential(
                Conv2dTranspose(352, 112, kernel_size=3, stride=2, padding=1, output_padding=1),
                Conv2d(112, 112, kernel_size=5, stride=1, padding=2, residual=True),
                Conv2d(112, 112, kernel_size=3, stride=1, padding=1, residual=True),
            )
            ]) 

        self.output_block = nn.Sequential(nn.Conv2d(176, output_block_channels, kernel_size=1, stride=1, padding=0))
        