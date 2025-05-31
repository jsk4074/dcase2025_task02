import torch
import torch.nn as nn
import torchvision.models as models 
import torchvision.transforms as transforms
import torch.nn.functional as F

import numpy as np

class conv_block(nn.Module):
    def __init__(self, channel_io):
        super(conv_block, self).__init__()

        self.channel_io = channel_io

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.channel_io[0], self.channel_io[1], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.channel_io[1]),
            nn.LeakyReLU(),
        )

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        c1 = self.conv1(x)
        output = self.max_pool(c1)

        return output

class transpose_block(nn.Module):
    def __init__(self, channel_input, channel_output):
        super(transpose_block, self).__init__()

        self.channel_input = channel_input
        self.channel_output = channel_output

        # self.decode_conv_module = nn.Sequential(
        #     nn.Conv2d(
        #         self.channel_input, 
        #         self.channel_input, 
        #         kernel_size=3, 
        #         stride=1, 
        #         padding=1, 
        #         bias=True
        #     ),
        #     # nn.BatchNorm2d(self.channel_input),
        #     nn.ReLU(),

        #     nn.Conv2d(
        #         self.channel_input, 
        #         self.channel_input, 
        #         kernel_size=3, 
        #         stride=1, 
        #         padding=1, 
        #         bias=True
        #     ),
        #     # nn.BatchNorm2d(self.channel_input),
        #     nn.ReLU(),
        # )

        self.transpose_decode_module = nn.Sequential(
            nn.ConvTranspose2d(
                self.channel_input, 
                channel_output, 
                kernel_size=3, 
                stride=2, 
                padding=1, 
                output_padding=1, 
                bias=False
            ),
            # nn.BatchNorm2d(channel_output),
            nn.LeakyReLU(),
        )

        self.transpose_decode_module_end = nn.Sequential(
            nn.ConvTranspose2d(
                self.channel_input, 
                channel_output, 
                kernel_size=3, 
                stride=2, 
                padding=1, 
                output_padding=1, 
                bias=False
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        if self.channel_output == 1: x = self.transpose_decode_module_end(x) 
        else: x = self.transpose_decode_module(x) 
        return x 

class decoder_module(nn.Module):
    def __init__(self, channel_size, fc_input_size):
        super(decoder_module, self).__init__()

        self.channel_size = channel_size

        self.conv1 = nn.Sequential(
            conv_block([1, channel_size[0]]),
            conv_block([channel_size[0], channel_size[1]]),
            conv_block([channel_size[1], channel_size[2]]),
            conv_block([channel_size[2], channel_size[3]]),
        )

        self.unflatten_size, self.fc_input_size = self._calculate_fc_input_size(fc_input_size)
        self.fc_decoder_embedding = nn.Sequential(
            nn.Linear(1024, 1024 * 2, bias=False), 
            nn.BatchNorm1d(1024 * 2),
            nn.LeakyReLU(),

            nn.Linear(1024 * 2, self.fc_input_size, bias=False), 
            nn.BatchNorm1d(self.fc_input_size),
            nn.LeakyReLU(),
        )

        self.transpose_decode_module_0 = transpose_block(self.channel_size[3], self.channel_size[2])
        self.transpose_decode_module_1 = transpose_block(self.channel_size[2], self.channel_size[1])
        self.transpose_decode_module_2 = transpose_block(self.channel_size[1], self.channel_size[0])
        self.transpose_decode_module_3 = transpose_block(self.channel_size[0], 1)

    def _calculate_fc_input_size(self, input_size):
        with torch.no_grad():
            x = torch.randn(1, 1, input_size, input_size)
            x = self.conv1(x)  
            return x, x.view(1, -1).shape[1]
        
    def forward(self, x):
        fc_decoded = self.fc_decoder_embedding(x)
        fc_decoded = fc_decoded.view(-1, self.channel_size[-1], self.unflatten_size.size()[3], self.unflatten_size.size()[3])

        d1 = self.transpose_decode_module_0(fc_decoded)
        d1 = self.transpose_decode_module_1(d1)
        d1 = self.transpose_decode_module_2(d1)
        d1 = self.transpose_decode_module_3(d1)

        return d1

class decoder(nn.Module):
    def __init__(self, input_size, ae_channel_size):
        super(decoder, self).__init__()
        self.decoder_module = decoder_module(ae_channel_size, input_size)

    def forward(self, encoded):
        x = self.decoder_module(encoded)
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        return x