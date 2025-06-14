import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

import random, os
import numpy as np
import wandb
from glob import glob 

seed = 7777
import warnings
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Custom files 
from architecture.encoder import encoder_model
from architecture.conv_decoder import conv_decoder
from architecture.decoder import decoder
from architecture.fcn import fcn

# Train functions 
from train.backbone import model_fit_encoder
from train.ascent import model_fit_anomaly_generator
from train.fcn import model_fit_discriminator

# Loss functions 
from usefulthings.ssim_loss import SSIMLoss

domain = ['source', 'target'] 
# class_names = ['ToyTrain', 'gearbox', 'ToyCar', 'bearing', 'valve', 'fan', 'slider']
class_names = [
    'AutoTrash',
    'BandSealer',
    'CoffeeGrinder',
    'HomeCamera',
    'Polisher',
    'ScrewFeeder',
    'ToyPet ToyRCCar',
]

def main(config = None): 
    eval_class_name = class_names[0]

    # Logging
    log_dir = "./runs/eval_" + eval_class_name + "_100epoch_resnet50"
    wandb.init(
        project="dcase_2025_t02_tb",
        dir = log_dir,  # Optional: makes log scanning easier
        sync_tensorboard=True,
    )
    writer = SummaryWriter(log_dir = log_dir) 

    dataset_path = "../data/unziped/add/" + eval_class_name + "/train/*" # DCASE Original
    print(dataset_path) 
    device = torch.device("cuda") 

    ######################### PARAMETER #########################

    config = wandb.config
    # General finetuning parameters 
    dataset_path = dataset_path
    epoch = 300
    batch_size = 64
    lr = config.lr

    # Model input wise
    input_size = 256
    # ae_channel_size = [32, 64, 128, 256]
    ae_channel_size = config.ae_channel_size
    latent_dim = 1024
    start_feature_size = 8

    # Loss functions
    descent_generator_criterion = SSIMLoss()
    ascent_generator_criterion = SSIMLoss()

    ##############################################################

    model_encoder = encoder_model(
        model_name="efficientnet_b0",
        # model_name="wide_resnet50",
        
        # model_name="vit_base_patch16_224",

        # model_name="resnet50",
        # model_name="resnet18",
        input_size=256,
        in_chans=1,          
        embedding_dim=1024
    ).to(device)

    model_descent_decoder = decoder(
        input_size = input_size, 
        ae_channel_size = ae_channel_size,
    ).to(device)

    # model_ascent_decoder = decoder(
    #     input_size = input_size, 
    #     ae_channel_size = ae_channel_size,
    # ).to(device)

    model_ascent_decoder = conv_decoder(
        latent_dim, 
        start_feature_size,
    ).to(device)

    model_discriminator = fcn().to(device)

    # torch.save(model_ascent_decoder, "./weights/model_ascent_decoder.pth")
    # torch.save(model_descent_decoder, "./weights/model_descent_decoder.pth")
    # torch.save(model_encoder, "./weights/model_descent_encoder.pth")

    # Backbone pretraining 
    model_encoder = model_fit_encoder(
        batch_size = batch_size, 
        learning_rate = lr, 
        epoch = epoch, 
        dataset_path = dataset_path,  
        model_encoder = model_encoder, 
        model_descent_decoder = model_descent_decoder, 
        mode = "train", 
        descent_generator_criterion = descent_generator_criterion, 
        device = device, 
        input_size = input_size, 
        writer = writer, 
    )

    # Anomaly generator
    model_encoder, ascent_model, normal_total_data, synth_anomaly_data = model_fit_anomaly_generator(
        batch_size = batch_size, 
        learning_rate = lr, 
        epoch = epoch, 
        dataset_path = dataset_path,  
        model_encoder = model_encoder, 
        model_ascent_decoder = model_ascent_decoder, 
        mode = "train", 
        ascent_generator_criterion = ascent_generator_criterion, 
        device = device, 
        input_size = 256, 
        writer = writer, 
    )

    # Discriminator training 
    model_encoder = model_fit_discriminator(
        batch_size = batch_size, 
        learning_rate = lr, 
        epoch = epoch, 
        dataset_path = dataset_path, 
        discriminator_train_data = synth_anomaly_data,
        normal_train_data = normal_total_data,
        model_encoder = model_encoder, 
        model_ascent_decoder = model_ascent_decoder, 
        model_discriminator = model_discriminator, 
        mode = "train", 
        ascent_generator_criterion = ascent_generator_criterion, 
        device = device, 
        input_size = 256, 
        writer = writer, 
    )

    writer.flush()
    writer.close()
    wandb.finish()

if __name__ == "__main__": 
    # mp.set_start_method('spawn', force=True)
    sweep_config = {
        'method': 'grid'
    }

    batch_size = 64
    lr = 1e-3
    # ae_channel_size = [32, 64, 128, 256]
    ae_channel_size = [64, 128, 256, 512]

    parameters_dict = {
        'lr': {
            'values': [1e-2, 1e-3, 1e-4]
        },
        'ae_channel_size': {
            'values': [
                [16, 32, 64, 128], 
                [32, 64, 128, 256], 
                [64, 128, 256, 512], 
                [128, 256, 512, 1024], 
            ]
        },
    }

    sweep_config['parameters'] = parameters_dict
    sweep_id = wandb.sweep(sweep_config, project="dcase_2025_t02_tb")

    wandb.agent(sweep_id, main, count=50)    
    


# print("=" * 25, "DEBUG", "=" * 25)
