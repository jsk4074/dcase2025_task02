import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def gaussian(window_size, sigma):
    """
    Creates a 1D Gaussian distribution with the given window_size and sigma.
    """
    gauss = torch.Tensor([
        math.exp(-((x - window_size // 2) ** 2) / float(2 * sigma ** 2))
        for x in range(window_size)
    ])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    """
    Creates a 2D Gaussian window that will be used to convolve with the image.
    The window is expanded to have 'channel' channels.
    """
    _1D_window = gaussian(window_size, sigma=1.5).unsqueeze(1)  # shape: [window_size, 1]
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    # shape: [1, 1, window_size, window_size]
    
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    # shape: [channel, 1, window_size, window_size]
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    """
    Internal function that computes the SSIM map between img1 and img2.
    Both tensors should be of shape [B, C, H, W].
    """

    # Convolve to get mu (local mean) for img1 and img2
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # Convolve to get local variance and covariance
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12   = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    # Constants for numerical stability (these are the standard ones for SSIM)
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # SSIM formula
    ssim_map = ((2.0 * mu1_mu2 + C1) * (2.0 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean() if size_average else ssim_map

def ssim(img1, img2, window_size=11, size_average=True):
    """
    Computes the mean SSIM between two batches of images.
    """
    # Expecting img1, img2: [B, C, H, W]
    (batch_size, channel, height, width) = img1.size()

    # Create the Gaussian window once based on the channel size
    window = create_window(window_size, channel).to(img1.device)

    return _ssim(img1, img2, window, window_size, channel, size_average)

class SSIMLoss(nn.Module):
    """
    A handy module that computes 1 - SSIM(img1, img2).
    Minimizing this means maximizing structural similarity.
    """
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average

    def forward(self, img1, img2):
        return 1.0 - ssim(img1, img2, self.window_size, self.size_average)
