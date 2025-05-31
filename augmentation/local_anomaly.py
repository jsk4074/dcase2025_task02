import torch
import torch.nn.functional as F
import numpy as np
import random
from torchvision import transforms
from glob import glob
import librosa

def audio2stft(raw_audio): 
    # Normalize input 0 to 1 
    raw_audio = raw_audio / np.max(raw_audio)
    # raw_audio = raw_audio / np.max(np.abs(raw_audio))

    # Turn raw audio to stft
    stft_data = librosa.stft(
        y = raw_audio,
        n_fft = 256,
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
    else: 
        audio_data = [raw_data[0], 1]

    # Cropping features to "crop_sec" + augment feature 
    sr = int(16e3) 
    
    # Need to improve on multi-view without maual tuning 
    audio_data = [
        np.array(audio_data[0][sr * padding : int(sr * (crop_sec + 1))]), 
        np.array(audio_data[0][int(sr * (crop_sec + 1) - 2) : int(sr * (crop_sec * 2 + 1) - 2)]), 
        np.array(audio_data[0][int(sr * (crop_sec + 1)) : int(sr * (crop_sec * 2 + 1))]), 
        audio_data[1], 
    ]

    # Extracting features
    # [audio_feature_1, audio_feature_2, audio_feature_3, label] 
    feature_data = [
        audio2stft(audio_data[0]),
        audio2stft(audio_data[1]),
        audio2stft(audio_data[2]),
        audio_data[3]
    ] 

    return feature_data

def generate_perlin_noise(size=(128, 128), scale=8):
    def f(x, y):
        return np.sin(2 * np.pi * x / scale) * np.sin(2 * np.pi * y / scale)

    x = np.linspace(0, size[0], size[0])
    y = np.linspace(0, size[1], size[1])
    xv, yv = np.meshgrid(x, y)
    
    noise = f(xv, yv)
    noise = (noise - noise.min()) / (noise.max() - noise.min())  # Normalize [0,1]
    return (noise > 0.5).astype(np.float32)  # Binarize the noise

def load_random_texture(input_size, dtd_path="../../data/unziped/dev/fan/train/"):
    texture_files = glob(dtd_path + "*.wav")
    texture_path = random.choice(texture_files)

    texture = torch.tensor(path2feature(texture_path, None)[0]).unsqueeze(dim = 0)

    transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        # transforms.ToTensor()
    ])
    
    return transform(texture)

def apply_local_anomaly(image, perlin_mask, texture, device, input_size, beta=0.2):
    image = F.interpolate(image.unsqueeze(0), size=(input_size, input_size), mode="bilinear", align_corners=False).squeeze(0)
    perlin_mask = torch.tensor(perlin_mask).unsqueeze(0).to(device)  # Convert to tensor
    
    # Normalize mask
    perlin_mask = perlin_mask.float()
    mask_inv = 1 - perlin_mask  # Inverse mask for background
    
    # Blend the normal image and texture
    synthetic_image = image * mask_inv + (1 - beta) * texture * perlin_mask + beta * image * perlin_mask
    return synthetic_image

def generate_synthetic_anomaly(
        feature, 
        dtd_path, 
        device, 
        input_size,
    ):
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        # transforms.ToTensor()
    ])
    
    image = transform(feature).to(device)
    
    # Generate Perlin mask
    perlin_mask = generate_perlin_noise(size=(input_size, input_size), scale=10)

    # Load a random texture
    texture = load_random_texture(input_size, dtd_path).to(device)

    # Generate synthetic anomaly
    beta = np.random.normal(0.5, 0.12)  # Sample β from N(0.5, 0.12)
    # beta = np.clip(beta, 0.2, 0.8)  # Keep β in [0.2, 0.8]
    beta = np.clip(beta, 0.2, 0.8)  # Keep β in [0.2, 0.8]

    synthetic_image = apply_local_anomaly(image, perlin_mask, texture, device, input_size, beta)
    
    return synthetic_image

