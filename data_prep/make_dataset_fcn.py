import torch
from torch.utils.data import Dataset
from torchaudio.transforms import FrequencyMasking, TimeMasking
from torchvision.transforms import RandomErasing

from augmentation.local_anomaly_stft import apply_spectral_anomaly
import numpy as np

torch.manual_seed(7777)
np.random.seed(7777)

# Make custom dataset 
class CustomDataset_fcn(Dataset): 
    def __init__(self, data = None, label = None):
        self.data = data 
        self.label_fc = label 

    # Data len return 
    def __len__(self): 
        return len(self.label_fc)

    # Index to data mapping 
    def __getitem__(self, idx): 
        stft_2d = self.data[idx]
        y = self.label_fc[idx]
        
        return stft_2d, y