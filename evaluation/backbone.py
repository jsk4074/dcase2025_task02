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

def model_fit_encoder(
        batch_size, 
        learning_rate, 
        epoch, 
        dataset_path,  
        model_encoder, 
        model_descent_decoder, 
        mode = "train", 
        descent_generator_criterion = None, 
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
    model_descent_decoder.to(device)

    # Model fit
    print("Start training ...")
    print("="*50)

    # Optimizer and Loss function defines
    optimizer_encoder = optim.AdamW(
        model_encoder.parameters(), 
        lr=learning_rate, 
        weight_decay=1e-5,
    )

    optimizer_descent_decoder = optim.AdamW(
        model_descent_decoder.parameters(), 
        lr=learning_rate,  
        weight_decay=1e-5,
    )

    scaler = GradScaler()
    print("="*25, "Ascent Descent model train starting", "="*25)

    for epoch_count in range(epoch): 
        # For decoded feature collection 
        decoded_descent_feature_total = None

        print("Epoch:", epoch_count + 1)
        train_loader_prog = tqdm(train_loader)
        
        step_counter = 0
        for img, _, label in train_loader_prog:
            img = img.to(device)
            label = label.to(device)

            optimizer_encoder.zero_grad()
            optimizer_descent_decoder.zero_grad()

            encoded_feature = model_encoder(img)

            decoded_descent_feature = model_descent_decoder(encoded_feature) 
            descent_loss = descent_generator_criterion(decoded_descent_feature, img) 

            scaler.scale(descent_loss).backward()
            for opt in (optimizer_encoder, optimizer_descent_decoder):
                scaler.step(opt)
            scaler.update()

            # Saving data for visualization  
            if decoded_descent_feature_total is None: 
                normal_total = img.detach().cpu()
                decoded_descent_feature_total = decoded_descent_feature.detach().cpu()
            else: 
                normal_total = torch.cat((normal_total, img.detach().cpu()), 0)
                decoded_descent_feature_total = torch.cat((decoded_descent_feature_total, decoded_descent_feature.detach().cpu()), 0)

            # tqdm
            train_loader_prog.set_description( 
                "epoch: %d, descent_loss: %.6f"
                %(
                    epoch_count + 1,
                    descent_loss.item(),
                ) 
            ) 
            step_counter += 1

            # Detach tensors after using them
            decoded_descent_feature = decoded_descent_feature.detach()
            img = img.detach()

        # Logging features 
        # Save output feature image via tensorboard
        make_grid_and_log(
            img, 
            writer, 
            'Original feature',
            epoch_count, 
        )

        make_grid_and_log(
            decoded_descent_feature, 
            writer, 
            'Descent feature',
            epoch_count, 
        )

        # Add scalar metric descent / ascent losses  
        writer.add_scalar("Descent Loss", descent_loss, epoch_count + 1)

        # For saving embedding for normal and abnormal features 
        decoded_descent_feature_total_view = decoded_descent_feature_total.view(len(decoded_descent_feature_total), -1).detach() 

        # Adding perlin noise label
        descent_embedding_metadata = torch.tensor([2 for i in range(len(decoded_descent_feature_total))]) 
        
        decoded_feature, embedding_metadata = select_n_random(
            torch.cat((
                decoded_descent_feature_total_view, 
            ), 0), 
            torch.cat((
                descent_embedding_metadata, 
            ), 0), 
        )

        # Every 25 (epoch inclouding epoch 0) logs embedding for PCA, UMAP ...
        if (epoch_count + 1) % 25 == 0: 
            # Model weight visualization
            print("Adding histogram for weights")
            log_weight_histograms(writer, model_encoder, model_name="Descent_Encoder", epoch=epoch_count)
            log_weight_histograms(writer, model_descent_decoder, model_name="Descent_Decoder", epoch=epoch_count)

            # Decoded embedding for EFA
            print("Adding embedding for checking distributions")
            writer.add_embedding(
                decoded_feature,
                metadata = list(embedding_metadata.detach()),
                global_step = str(epoch_count) + "embedding",
            )

        # Accuracy calculation for batch
        print("Train descent Loss:", descent_loss.item())
        print("="*50)

    return model_encoder


# print("=" * 25, "DEBUG", "=" * 25)