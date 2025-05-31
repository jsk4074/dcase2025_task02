import torch
import torch.nn as nn
import torchvision.models as models 
import torchvision.transforms as transforms
import torch.nn.functional as F

import numpy as np

class upsample_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(upsample_block, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)

class conv_decoder(nn.Module):
    def __init__(self, latent_dim, start_feature_size):
        super(conv_decoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, start_feature_size * start_feature_size * 1024),
            nn.ReLU(),
        )
        self.start_feature_size = start_feature_size

        # self.decode = nn.Sequential(
        #     upsample_block(256, 128),
        #     upsample_block(128, 64),
        #     upsample_block(64, 32),
        #     upsample_block(32, 16),
        #     nn.Conv2d(16, 1, kernel_size=1),
        #     nn.Sigmoid()
        # )

        self.decode = nn.Sequential(
            # upsample_block(1024, 512),
            # upsample_block(512, 256),
            # upsample_block(256, 128),
            upsample_block(1024, 64),
            upsample_block(64, 32),
            upsample_block(32, 16),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 1024, self.start_feature_size, self.start_feature_size)
        x = self.decode(x)
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)

        return x
