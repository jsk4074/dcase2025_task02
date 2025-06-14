import torch
import torch.nn as nn
import torch.optim as optim
from torchviz import make_dot
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from torchaudio.transforms import FrequencyMasking, TimeMasking
from torchvision.transforms import RandomErasing


import librosa 
import numpy as np
from tqdm import tqdm 
from glob import glob 

from random import randint

from augmentation.local_anomaly_stft import apply_spectral_anomaly

torch.manual_seed(7777)
np.random.seed(7777)

def audio2stft(raw_audio): 
    # Normalize input 0 to 1 
    # raw_audio = raw_audio / np.max(raw_audio)
    # raw_audio = raw_audio / np.max(np.abs(raw_audio))

    # Turn raw audio to stft
    stft_data = librosa.stft(
        y = raw_audio,
        n_fft = 512,
        hop_length = 256,
        window='hann',
    )

    # Get absolute value for stft data
    stft_data = abs(stft_data)

    # Convert stft to Log-Scaling
    stft_data = librosa.amplitude_to_db(
        stft_data, 
        ref=np.max
    )

    # Normalize Log-Scaling data
    stft_data = (stft_data - np.min(stft_data)) / (np.max(stft_data) - np.min(stft_data))

    # Crop out empty bends 
    stft_data = stft_data[:180, :]
    
    return stft_data

def path2feature(path, class_names, padding = 1, crop_sec = 3): 
    domain = ['source', 'target']

    # Check for empty list and things ...
    if len(path) == 0: 
        print("#"*20, "ERROR", "#"*20)
        print("No file has been found")
        print("#"*20, "ERROR", "#"*20)
        return None
    
    # Loading data [audio_wav, label]
    raw_data = [librosa.load(path, sr=16e3)[0], path.split("/")[-3], path.split("_")[2], path.split("_")[4]]

    # label and domain encoding to int [audio_feature, label] 
    if raw_data[3] == "normal":
        audio_data = [raw_data[0], 0]
    elif raw_data[3] == "anomaly": 
        audio_data = [raw_data[0], 1]
    else: return print("ERRRRROR_DATALODAER_NO_LABEL")

    # Cropping features to "crop_sec" + augment feature 
    sr = int(16e3) 
    
    # Extracting features
    stft_data = audio2stft(audio_data[0])

    labels = audio_data[1]

    return stft_data, labels

class CustomDataset(Dataset): 
    def __init__(self, dataset_root_path, transform=None, crop=False, feature_extraction=False, mode="train", resize_shape=(128, 128), device="cpu"):
        self.transform = transform
        self.resize_shape = resize_shape
        self.mode = mode 
        self.device = device

        # Using sorted() to keep order consistent
        if mode in ["train", "eval"]:
            self.dataset_path = sorted(glob(dataset_root_path))
        elif mode == "test": 
            self.dataset_path = sorted(glob(dataset_root_path.replace("train", "test")))
        elif mode == "eval": 
            self.dataset_path = sorted(glob(dataset_root_path.replace("train", "test")))
            self.dataset_path = sorted(glob(self.dataset_path.replace("add", "eval_2025")))

            print(self.dataset_path)
            print(self.dataset_path)
            print(self.dataset_path)
        else:
            raise ValueError(f"Unknown mode: {mode}. Choose from 'train', 'test', or 'eval'.")
        
        print("="*10, "LOADING DATASET", "="*10)
        self.dataset_stft = []
        self.dataset_labes = []
        for indi_path in tqdm(self.dataset_path):
            stft_data, labels = path2feature(
                path=indi_path,
                class_names="", 
                padding=1,
                crop_sec=3,
            )

            self.dataset_stft += [stft_data]
            self.dataset_labes += [labels]

    def __len__(self): 
        return len(self.dataset_stft)

    def __getitem__(self, idx):         
        # stft data processing
        stft_data = self.dataset_stft[idx]
        stft_data = torch.FloatTensor(stft_data).unsqueeze(0)
        stft_data = torch.nn.functional.interpolate(
            stft_data.unsqueeze(0), size=self.resize_shape, mode="bilinear", align_corners=False
        ).squeeze(0)
 

        label = self.dataset_labes[idx]
        label = torch.tensor(label)  

        # Apply augmentation only for the 'train' mode
        if self.mode == "train":
            return stft_data, stft_data, label
        
        elif self.mode == "test":
            return stft_data, label
        
        elif self.mode == "eval":
            return stft_data
        
        else: 
            raise ValueError("Invalid mode. Choose from 'train' or 'test'.")

