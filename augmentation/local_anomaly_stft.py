import torch
import torch.nn.functional as F

import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from augmentation.perlin import rand_perlin_2d_octaves

# def gaussian_blur_tensor(tensor, sigma=0.1, kernel_size=3):    
#     channels = tensor.shape[1]
#     # Create a 1D Gaussian kernel.
#     coords = torch.arange(kernel_size, dtype=tensor.dtype, device=tensor.device) - (kernel_size - 1) / 2.0
#     g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
#     g = g / g.sum()
    
#     # Create 2D Gaussian kernel using the outer product.
#     g_kernel = torch.outer(g, g)  # shape: [kernel_size, kernel_size]
    
#     # Reshape kernel to [channels, 1, kernel_size, kernel_size].
#     g_kernel = g_kernel.expand(channels, 1, kernel_size, kernel_size)
    
#     # Use groups=channels if more than one channel, else groups=1.
#     groups = channels if channels > 1 else 1
    
#     # Apply convolution with appropriate padding.
#     blurred = F.conv2d(tensor, weight=g_kernel, groups=groups, padding=kernel_size // 2)
#     return blurred

def apply_spectral_anomaly(
        spectrogram, 
        anomaly_strength = 0.5, 
        anomaly_region = (0., 1.), 
        input_size = 128, 
        device = None
    ):

    synthetic_spectrogram = spectrogram.detach().clone().to(device)

    # Define affected region
    freq_range, time_range = input_size[0], input_size[1]
    freq_start = int(anomaly_region[0] * freq_range)
    freq_end = int(anomaly_region[1] * freq_range)
    time_start = int(anomaly_region[0] * time_range)
    time_end = int(anomaly_region[1] * time_range)

    # Generate Perlin noise for structured distortion
    # perlin_noise = (rand_perlin_2d_octaves(size=(input_size[0], input_size[1]), scale=8) * anomaly_strength)
    perlin_noise = rand_perlin_2d_octaves((input_size[0], input_size[0]), (4, 4), 5, device=device)
    ########## NORMALIZE ########## 
    perlin_noise = perlin_noise / torch.max(perlin_noise)
    ########## NORMALIZE ########## 
    perlin_noise = perlin_noise.unsqueeze(0) * anomaly_strength

    # Apply structured noise in the selected region
    synthetic_spectrogram[:, freq_start:freq_end, time_start:time_end] -= perlin_noise[:, freq_start:freq_end, time_start:time_end]

    # # Apply Frequency Band Dropout (simulates missing frequencies)
    # if random.random() > 0.5:  # 50% chance of applying dropout
    #     band_start = random.randint(0, input_size[0])  # Randomly pick a frequency band
    #     band_width = random.randint(1, 12)  # Random band width
    #     # synthetic_spectrogram[:, band_start:band_start + band_width, :] *= (0.5 + 0.5 * torch.rand(1))  # Reduce intensity
    #     dropout_factor = 0.5 + 0.5 * torch.rand(1, device=device)
    #     synthetic_spectrogram[:, band_start:band_start + band_width, :] *= dropout_factor


    # Apply Gaussian blur (simulates signal smearing)
    # synthetic_spectrogram = torch.tensor(gaussian_filter(synthetic_spectrogram.to("cpu").numpy(), sigma=0.5)).squeeze().unsqueeze(0)
    # synthetic_spectrogram = torch.tensor(synthetic_spectrogram.to("cpu"))
    synthetic_spectrogram = synthetic_spectrogram.detach().cpu().clone()

    # Apply Gaussian blur 
    # synthetic_spectrogram = gaussian_blur_tensor(synthetic_spectrogram.unsqueeze(0), sigma=0.5, kernel_size=3)
    
    # Remove the added batch dimension if not needed.
    # synthetic_spectrogram = synthetic_spectrogram.squeeze(0)

    return synthetic_spectrogram
    # return torch.abs(synthetic_spectrogram / torch.max(synthetic_spectrogram))
