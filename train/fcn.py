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
from train.validation import model_test
from data_prep.make_dataset_fcn import CustomDataset_fcn
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

    # writer.add_image(tag, grid_image, epoch_count)

    log_image(writer, tag, grid_image, epoch_count)

def dataset_prep_fcn(data = None, label = None, batch_size = None):
    dataset = CustomDataset_fcn(
        data,
        label,
    )

    train_loader = DataLoader( 
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,  
        # persistent_workers=True,
        # pin_memory=True,  
    ) 
    
    return train_loader

def model_fit_discriminator(
        batch_size, 
        learning_rate, 
        epoch, 
        dataset_path, 
        discriminator_train_data,
        normal_train_data,
        model_encoder, 
        model_ascent_decoder, 
        model_discriminator, 
        mode, 
        ascent_generator_criterion, 
        device, 
        input_size, 
        writer, 
    ):

    print("============================ FCN (Discriminator) model Train mode ============================")

    print(normal_train_data.size())
    print(normal_train_data.size())
    print(normal_train_data.size())
    print(discriminator_train_data.size())
    print(discriminator_train_data.size())
    print(discriminator_train_data.size())

    # Load dataset
    normal_train_data = normal_train_data.detach().to(device)
    normal_label = torch.tensor([0 for i in range(len(normal_train_data))]).to(device)
    anomaly_label = torch.tensor([1 for i in range(len(discriminator_train_data))]).to(device)

    fcn_train_data = torch.cat((
        discriminator_train_data,
        normal_train_data,
    ), 0)

    label = torch.cat((
        anomaly_label,
        normal_label,
    ), 0)
    
    print(fcn_train_data.size())
    print(label.size())

    print("Loading FCN dataset ...")
    fcn_train_loader = dataset_prep_fcn(
        fcn_train_data, 
        label,
        batch_size, 
    )

    # Auto config model to fit w/ gpu or cpu 
    print("Training on :", device)
    model_encoder.to(device).train()
    model_discriminator.to(device).train()

    # Model fit
    print("Start FCN training ...")
    print("="*50)

    # Freeze encoder 
    for param in model_encoder.parameters(): 
        param.requires_grad = False


    # Optimizer and Loss function defines
    optimizer_fcn = optim.AdamW(model_discriminator.parameters(), lr=learning_rate * 1e-2, weight_decay=1e-5)

    # Loss Functions 
    criterion = ops.sigmoid_focal_loss
    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCELoss()
    # criterion = nn.HingeEmbeddingLoss()
    # criterion = nn.MSELoss()

    # model_encoder.to(device).eval()
    for epoch_count in range(int(epoch)):
        model_encoder.to(device).train()
        model_discriminator.to(device).train()
        
        print("Epoch:", epoch_count + 1)
        fcn_train_loader_prog = tqdm(fcn_train_loader)
        
        step_counter = 0
        fcn_loss_total = 0
        correct_predictions = 0
        total_samples = 0

        embedding_feature_total = None
        for img, label in fcn_train_loader_prog:
            img = img.to(device)
            label = label.to(device)
            bce_label = torch.nn.functional.one_hot(label, num_classes=2).float().to(device)

            # Fitting FCN model 
            encoder_fearute = model_encoder(img)
            # model_output = model_discriminator(encoder_fearute.detach()) 
            model_output = model_discriminator(encoder_fearute) 

            # Loss calculation
            fcn_loss = criterion(model_output, bce_label).mean()
            # fcn_loss = criterion(model_output, label) 

            # optimizer_descent_encoder.zero_grad()  
            optimizer_fcn.zero_grad()  
            fcn_loss.backward()
            optimizer_fcn.step()

            fcn_loss_total += fcn_loss.item()
            step_counter += 1

            # Compute Accuracy
            preds = torch.argmax(model_output, dim=1)  
            correct_predictions += (preds == label).sum().item()  
            total_samples += label.size(0)  

            # Logging embeddings 
            if embedding_feature_total is None: 
                embedding_feature_total = encoder_fearute.detach().cpu()
                embedding_label_total = label.detach().cpu()
            else: 
                embedding_feature_total = torch.cat((embedding_feature_total.detach().cpu(), encoder_fearute.detach().cpu()))
                embedding_label_total = torch.cat((embedding_label_total.detach().cpu(), label.detach().cpu()))
            
            # tqdm
            fcn_train_loader_prog.set_description(
                "epoch: %d, fc_loss: %.6f, ACC: %.6f" 
                % (
                    epoch_count + 1,
                    fcn_loss.item(),
                    (correct_predictions / total_samples) * 100,
                )
            )
            bce_label = []

        log_weight_histograms(writer, model_discriminator, model_name="Discriminator", epoch=epoch_count)
        writer.add_scalar("FCN Loss", fcn_loss, epoch_count + 1)
        writer.add_scalar("FCN Accuracy", (correct_predictions / total_samples) * 100, epoch_count + 1)

        embedding_feature, embedding_label = select_n_random(
            torch.cat((
                embedding_feature_total, 
            ), 0), embedding_label_total
        )

        print("Train FCN Loss:", fcn_loss_total / (int(epoch) * len(img))) 
        print("Train FCN Accuracy: {:.2f}%".format((correct_predictions / total_samples) * 100))
        print("="*50)

        test_path = dataset_path.replace("train", "test")
        print(test_path)
        
        model_encoder.eval()
        model_discriminator.eval()

        # model_ascent_encoder.eval()
        with torch.no_grad():
            model_test(
                batch_size = 1, 
                dataset_path = test_path,
                model_fe = model_encoder, 
                model_fcn = model_discriminator, 
                device = device,
                resize_shape = input_size,
                writer = writer,
                epoch = epoch_count,
                train_embedding_feature = embedding_feature, 
                train_embedding_label = embedding_label,
            )    

# print("=" * 25, "DEBUG", "=" * 25)