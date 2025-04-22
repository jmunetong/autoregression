import os
import sys

from accelerate import Accelerator
import wandb
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn, optim
import einops
from transformers import  ViTImageProcessor
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader

from utils import get_device
from data_preprocessing import XrdDataset


FEATURE_EXTRACTOR_PATH = "google/vit-base-patch16-224"
DATA_PATH = "data"
URL_MODEL = "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors"
NUM_DETECTORS = 40



def run(args):
    beta = args.beta
    run_logger = wandb.init(project="vae_kl_tunning", config=args)
    feature_extractor = ViTImageProcessor.from_pretrained(FEATURE_EXTRACTOR_PATH)
    dataset = XrdDataset(data_dir=args.data_path,    feature_extractor=feature_extractor)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    device = get_device()
    print(f'Current CUDA device is:{device}')
    vae = AutoencoderKL.from_single_file(URL_MODEL)
    # vae.to(device)
    vae.train()
    optimizer = optim.Adam(vae.parameters(), lr=1e-4)
    accelerator = Accelerator()
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    vae, optimizer, dataloader, lr_scheduler = accelerator.prepare(
    vae, optimizer, dataloader, lr_scheduler
)
    total_params = sum(p.numel() for p in vae.parameters())
    print(f'Total parameters: {total_params:,}')
    best_loss = float('inf')
    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        epoch_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_recon_loss = 0.0
        for _, batch in enumerate(dataloader):
            optimizer.zero_grad()

            ## Encoding Step
            posterior = vae.encode(batch).latent_dist
            mu_posterior = posterior.mean
            logvar_posterior = posterior.logvar

            # Decoding step
            recon_i = vae.decode(posterior.sample()).sample
            
            # Loss Function Computation
            kl_loss_i = -0.5 * torch.sum(1 + logvar_posterior - mu_posterior.pow(2) - torch.exp(logvar_posterior))
            kl_loss_i /= batch.size(0)
            recon_loss_i = F.mse_loss(recon_i, batch, reduction='mean')
            loss_i = beta * recon_loss_i + kl_loss_i
            
            accelerator.backward(loss_i)
            
            # Track metrics
            batch_loss += loss_i.item()
            batch_kl_loss += kl_loss_i.item()
            batch_recon_loss += recon_loss_i.item()
            
            # Step optimizer after accumulating gradients
            optimizer.step()
            
        # Update epoch metrics with batch averages
        lr_scheduler.step()
        epoch_loss = batch_loss / len(dataloader)
        epoch_kl_loss = batch_kl_loss/len(dataloader)
        epoch_recon_loss = batch_recon_loss/ len(dataloader)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            print(f"New best loss: {best_loss}")
            # Save the model
            torch.save(vae.state_dict(), "vae_model.pth")
            try:
                vae.save_pretrained("vae_model")
                feature_extractor.save_pretrained("feature_extractor")
            except Exception as e:
                print(f"Error saving feature extractor: {e}")
    

        print(f"Epoch {epoch+1}, Loss: {epoch_loss}")
        run_logger.log({"epoch": epoch+1, "loss": epoch_loss, "recon_loss": epoch_recon_loss, "kl_loss": epoch_kl_loss})

        
if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default=DATA_PATH, help="Path to the data directory")
    parser.add_argument("--batch_size", "-b", type=int, default=8, help="Batch size for training")
    parser.add_argument("--detectors_per_batch", type=int, default=8, help="Number of detectors per batch")
    parser.add_argument("--num_epochs", "-e", type=int, default=20, help="Number of epochs for training")
    parser.add_argument("--beta", type=float, default=0.1, help="weight MSE Loss")
    args = parser.parse_args()
    run(args)