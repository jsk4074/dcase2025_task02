import torch
import torch.nn as nn
import torchvision.models as models 
import torchvision.transforms as transforms

import numpy as np

class fcn(nn.Module):
    def __init__(self, ):
        super(fcn, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(1024, 1024, bias = False), 
            nn.BatchNorm1d(1024),
            nn.ReLU(), 
            nn.Dropout(p=0.2),

            nn.Linear(1024, 512, bias = False), 
            nn.BatchNorm1d(512),
            nn.ReLU(), 
            nn.Dropout(p=0.2),
            
            nn.Linear(512, 2, bias = False), 
            nn.Sigmoid(), 
        )

    def forward(self, feature):
        output = self.fc(feature)

        return output