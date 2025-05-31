import torch
from augmentation.perlin import rand_perlin_2d_octaves

def mask2anomaly(anomaly_mask, normal_audio, device = "cuda"):
    # Assume img is your 2D image tensor (H, W)
    img = anomaly_mask.squeeze(0)  # (H, W)

    # Set a threshold (baseline is 0.1)
    threshold = 0.8

    # Binarize: pixels >= threshold -> 1, pixels < threshold -> 0
    binary_img = (img > threshold).float()

    perlin_noise = rand_perlin_2d_octaves((256, 256), (8, 8), 5, device=device)
    perlin_noise = torch.abs(perlin_noise) / torch.max(perlin_noise)
    perlin_noise = perlin_noise * 0.1

    mask_anomaly = binary_img * perlin_noise

    # mask_anomaly = img * threshold
    synth_anomaly = normal_audio + mask_anomaly

    return synth_anomaly / torch.max(synth_anomaly), binary_img







