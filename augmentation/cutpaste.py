import torch

def cutpaste(a,b,patch_size=(16,16)):
    B, C, H, W = a.shape
    x = torch.randint(0, H-patch_size[0], (1,)).item()
    y = torch.randint(0, W-patch_size[1], (1,)).item()
    a[:,:, x:x+patch_size[0], y:y+patch_size[1]] = \
       b[:,:, x:x+patch_size[0], y:y+patch_size[1]]
    
    return a
