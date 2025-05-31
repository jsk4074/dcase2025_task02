# Torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import make_grid
from  torchvision import ops 

# Normal operation
import numpy as np
from tqdm import tqdm  
from rich import print
import matplotlib.pyplot as plt

# Custom defined functions
from data_prep.make_dataset import CustomDataset
from usefulthings.usefulthings import log_image
from usefulthings.usefulthings import select_n_random
from usefulthings.usefulthings import log_weight_histograms
from augmentation.mask2anomaly import mask2anomaly

def make_grid_and_log(
        batch_images, 
        writer, 
        tag,
        epoch_count, 
    ): 

    grid_image = make_grid(
        batch_images[:4], 
        nrow=2,            # 3 images per row
        padding=2,         # pixels between images
        normalize=True,    # scale each image to [0,1]
        value_range=(0,1)  # if your data is already in [0,1]
    )

    # writer.add_image(tag, grid_image, epoch_count)

    log_image(writer, tag, grid_image, epoch_count)

def dataset_prep(path, batch_size, mode="train", resize_shape=(128, 128), device = None):
    dataset = CustomDataset(
        dataset_root_path = path, 
        transform = None, 
        crop = False, 
        feature_extraction = False, 
        mode = mode,
        resize_shape = resize_shape, 
        device = device,
    )

    train_loader = DataLoader( 
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,  
        # persistent_workers=True,
        pin_memory=True,  
    ) 
    
    return train_loader

def model_fit_anomaly_generator(
        batch_size, 
        learning_rate, 
        epoch, 
        dataset_path,  
        model_encoder, 
        model_ascent_decoder,
        mode = "train", 
        ascent_generator_criterion = None, 
        device = torch.device("cuda"), 
        input_size = 128, 
        writer = None, 
    ):


    # Load dataset
    print("Loading dataset ...")
    print("Loading path :", dataset_path)
    train_loader = dataset_prep(
        dataset_path, 
        batch_size, 
        mode = mode, 
        resize_shape=(input_size, input_size),
        device = device,
    )

    # Auto config model to fit w/ gpu or cpu 
    print("Training on :", device)
    model_encoder.to(device)

    # Model fit
    print("Start training ...")
    print("="*50)

    # Optimizer and Loss function defines
    optimizer_encoder = optim.AdamW(
        model_encoder.parameters(), 
        lr = learning_rate, 
        weight_decay = 1e-5,
    )

    optimizer_ascent_decoder = optim.AdamW(
        model_ascent_decoder.parameters(), 
        lr = learning_rate, 
        weight_decay = 1e-5, 
        maximize = True, 
    )

    scaler = GradScaler()
    print("="*25, "Ascent model train starting", "="*25)
    for param in model_encoder.parameters(): 
        param.requires_grad = False

    for epoch_count in range(epoch): 
        # For decoded feature collection 
        decoded_ascent_feature_total = None
        synth_anomaly_total = None
        binarized_mask_total = None

        print("Epoch:", epoch_count + 1)
        train_loader_prog = tqdm(train_loader)
        
        step_counter = 0
        for img, _, label in train_loader_prog:
            img = img.to(device)
            label = label.to(device)

            optimizer_encoder.zero_grad()
            optimizer_ascent_decoder.zero_grad()

            encoded_feature = model_encoder(img).detach() 

            # decoded_ascent_feature = model_ascent_decoder(encoded_feature.detach().requires_grad_(True)) 
            # ascent_loss = ascent_generator_criterion(decoded_ascent_feature, img) 

            # add Gaussian noise to the encoded feature
            # gaussian_noise = encoded_feature + torch.randn_like(encoded_feature) * 0.015
            # gaussian_noise = gaussian_noise.detach().requires_grad_(True)
            gaussian_noise = encoded_feature.detach().requires_grad_(True)

            # decode and compute ascent loss
            y_gaussian_noise = model_ascent_decoder(gaussian_noise)
            loss_gaussian_noise = ascent_generator_criterion(y_gaussian_noise, img)

            # compute gaussian_noise loss
            grad_gaussian = torch.autograd.grad(loss_gaussian_noise, gaussian_noise)[0] # shape [B,C,H,W]

            # compute per-sample gradient norm
            grad_gaussian_flat = grad_gaussian.view(grad_gaussian.size(0), -1) # [B, D]
            grad_norm = grad_gaussian_flat.norm(dim=1, keepdim=True) # [B,1]
            # expand back to [B,C,H,W] so division broadcasts correctly
            grad_norm_broadcast = grad_norm.view(grad_gaussian.size(0), *([1] * (grad_gaussian.dim() - 1)))

            # gradient-ascent step
            step_size = 1e-3
            gaussian_noise = gaussian_noise + step_size * grad_gaussian / (grad_norm_broadcast + 1e-8)

            # truncated projection into [r1, r2] = [0.5, 1.5]
            # r1, r2 = .6, .9
            r1, r2 = 45., 60.
            encoder_gaussian_diff = gaussian_noise - encoded_feature # [B,C,H,W]
            encoder_gaussian_diff_flat = encoder_gaussian_diff.view(encoder_gaussian_diff.size(0), -1)
            diff_norm = encoder_gaussian_diff_flat.norm(dim=1, keepdim=True)  # [B,1]
            diff_norm_broadcast = diff_norm.view(encoder_gaussian_diff.size(0), *([1] * (encoder_gaussian_diff.dim() - 1)))
            unit_diff = encoder_gaussian_diff / (diff_norm_broadcast + 1e-8)
            diff_norm_clipped = torch.clamp(diff_norm, min=r1, max=r2).view_as(diff_norm_broadcast)
            gaussian_noise = encoded_feature + unit_diff * diff_norm_clipped

            # decode the final ascent feature
            decoded_ascent_feature = model_ascent_decoder(encoded_feature)
            ascent_loss = ascent_generator_criterion(decoded_ascent_feature, img)

            for index, anomaly_mask in enumerate(decoded_ascent_feature): 
                anomaly_single, binarized_mask_single = mask2anomaly(anomaly_mask.detach(), img[index], device)
                # anomaly_single = cutpaste(anomaly_single.unsqueeze(0), img[index].unsqueeze(0))
                binarized_mask_single = binarized_mask_single.unsqueeze(0)

                if synth_anomaly_total == None and binarized_mask_total == None:
                    synth_anomaly_total = anomaly_single.unsqueeze(0)
                    binarized_mask_total = binarized_mask_single
                else:
                    synth_anomaly_total = torch.cat((synth_anomaly_total, anomaly_single.unsqueeze(0)), 0)
                    binarized_mask_total = torch.cat((binarized_mask_total, binarized_mask_single), 0)
            
            scaler.scale(ascent_loss).backward()
            scaler.step(optimizer_ascent_decoder)
            scaler.update()

            # Saving data for visualization  
            if decoded_ascent_feature_total is None: 
                normal_total = img.detach().cpu()
                decoded_ascent_feature_total = decoded_ascent_feature.detach().cpu()
            else: 
                normal_total = torch.cat((normal_total, img.detach().cpu()), 0)
                decoded_ascent_feature_total = torch.cat((decoded_ascent_feature_total, decoded_ascent_feature.detach().cpu()), 0)

            # tqdm
            train_loader_prog.set_description( 
                "epoch: %d, ascent_loss: %.6f"
                %(
                    epoch_count + 1,
                    ascent_loss.item(),
                ) 
            ) 
            step_counter += 1

            # Detach tensors after using them
            img = img.detach()

        # Logging features 
        # Save output feature image via tensorboard
        make_grid_and_log(
            decoded_ascent_feature, 
            writer, 
            'Ascent feature',
            epoch_count, 
        )
        make_grid_and_log(
            synth_anomaly_total, 
            writer, 
            'Synth anomaly',
            epoch_count, 
        )
        make_grid_and_log(
            binarized_mask_total, 
            writer, 
            'Binarized mask',
            epoch_count, 
        )

        # Add scalar metric ascent losses  
        writer.add_scalar("Ascent Loss", ascent_loss, epoch_count + 1)

        # For saving embedding for normal and abnormal features 
        decoded_ascent_feature_total_view = decoded_ascent_feature_total.view(len(decoded_ascent_feature_total), -1).detach() 

        # Adding perlin noise label
        ascent_embedding_metadata = torch.tensor([3 for i in range(len(decoded_ascent_feature_total))]) 
        
        decoded_feature, embedding_metadata = select_n_random(
            torch.cat((
                decoded_ascent_feature_total_view,
            ), 0), 
            torch.cat((
                ascent_embedding_metadata, 
            ), 0), 
        )

        # Every 25 (epoch inclouding epoch 0) logs embedding for PCA, UMAP ...
        if (epoch_count + 1) % 25 == 0: 
            # Model weight visualization
            print("Adding histogram for weights")
            log_weight_histograms(writer, model_ascent_decoder, model_name="Ascent_Decoder", epoch=epoch_count)

            # writer.add_histogram(
            #     'GAS/raw_step_magnitude',   # tag
            #     diff_norm.detach().cpu(),   # 1D tensor of your B norms
            #     global_step = epoch_count,     # or `step_counter`
            # )

            # Decoded embedding for EFA
            print("Adding embedding for checking distributions")
            writer.add_embedding(
                decoded_feature,
                metadata = list(embedding_metadata.detach()),
                global_step = str(epoch_count) + "embedding",
            )

        # Accuracy calculation for batch
        print("Train ascent Loss:", ascent_loss.item())
        print("="*50)

    # torch.save(decoded_ascent_feature_total, "./saved_features/decoded_ascent_tensor.pt")

    return model_encoder, model_ascent_decoder, normal_total, synth_anomaly_total
     