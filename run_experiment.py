import os
import sys

from tqdm import tqdm
from accelerate import Accelerator
import wandb

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader

from transformers import get_cosine_schedule_with_warmup
from diffusers import AutoencoderKL

from utils import get_device
from data_preprocessing import XrdDataset


FEATURE_EXTRACTOR_PATH = "google/vit-base-patch16-224"
DATA_PATH = "data"
URL_MODEL = "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors"

EXPERIMENTS = {
    422: 'mfxl1025422',
    522: 'mfxl1027522'
}

RECONS_LOSS = {
    "mse": nn.MSELoss(reduction="mean"),
    "l1": nn.L1Loss(reduction="mean")}

MODELS= {"vae_kl": AutoencoderKL}

def vae_config_dict():
    vae_config = {
        "in_channels": 1,
        "out_channels": 1,
        "latent_channels": 4,
        "down_block_types": ("DownEncoderBlock2D",) * 4,
        "up_block_types": ("UpDecoderBlock2D",) * 4,
        "block_out_channels": (64, 128, 256, 512),
        "sample_size": 64
    }
    return vae_config

def build_experiment_metadata(args):
    metadata = {
        "model_name": args.model_name,
        "data_id": args.data_id,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "beta_recons": args.beta_recons,
        "recons_loss": args.recons_loss,
        "input_shape": None,
        "latent_shape": None,
        "avg_pooling": args.avg_pooling,
        "weight_decay": args.weight_decay,
        "learning_rate": args.lr,
    }
    return metadata


def run(args):
    # TODO: ADD EXPERIMENT ID INFORMATION
    beta_recons = args.beta_recons
    experiment_dict = build_experiment_metadata(args)
    torch.cuda.empty_cache()
    print(EXPERIMENTS[args.data_id])
    
    # Dataset and Dataloader
    dataset = XrdDataset(data_dir=args.data_path,apply_pooling=args.avg_pooling, data_id=EXPERIMENTS[args.data_id])
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, )
    device = get_device()
    
    print(f'Current CUDA device is:{device}')
    
    # Model Instantiation
    vae_config = vae_config_dict()
    vae = MODELS[args.model_name](**vae_config)
    vae.train()
    # Optimizer
    num_training_steps = len(dataloader) * args.num_epochs
    num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup
    optimizer = optim.AdamW(vae.parameters(), lr=args.lr, weight_decay= args.weight_decay)

    # Scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps)
    
    # Accelerator instantiation
    accelerator = Accelerator()
    vae, optimizer, dataloader, scheduler = accelerator.prepare(
    vae, optimizer, dataloader, scheduler)
    vae = vae.module if hasattr(vae, "module") else vae
    recons_loss = RECONS_LOSS[args.recons_loss]

    total_params = sum(p.numel() for p in vae.parameters())
    print(f'Total parameters: {total_params:,}')
    
    best_loss = float('inf')
    
    # Loading weights and biases
    run_logger = wandb.init(project="vae_kl_tunning", config=args)
    
    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        
        epoch_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_recon_loss = 0.0
        s = None
        for i, batch in enumerate(dataloader):
            s = batch.shape[1:] if i ==0 else s
            if i==0:
                print(batch.shape)
            assert s == batch.shape[1:], f"Batch shape mismatch: {s} != {batch.shape[1:]}"
        sys.exit()

        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Training"):
            # Important before starting one forward pass
            optimizer.zero_grad()

            batch = batch.contiguous()
            if i == 0 and epoch == 0 and accelerator.is_main_process:
                print(f"Batch shape: {batch.shape}")
            ## Encoding Step
            print(f"Batch shape: {batch.shape}")
            posterior = vae.encode(batch).latent_dist
            mu_posterior = posterior.mean
            logvar_posterior = posterior.logvar
            
            # Decoding step
            posterior_sample = posterior.sample()
            if i == 0 and epoch == 0 and accelerator.is_main_process:
                experiment_dict["input_shape"] = batch.shape[1:]
                experiment_dict["latent_shape"] = posterior_sample.shape[1:]
                print(f"Batch shape: {batch.shape}")
                print(f"Posterior sample shape: {posterior_sample.shape}")
            
            recon_i = vae.decode(posterior_sample).sample
            
            # Loss Function Computation
            kl_loss_i = -0.5 * torch.sum(1 + logvar_posterior - mu_posterior.pow(2) - torch.exp(logvar_posterior))
            kl_loss_i /= batch.size(0)
            recon_loss_i = recons_loss(recon_i, batch)
            loss_i = beta_recons * recon_loss_i + kl_loss_i
            
            accelerator.backward(loss_i)
            
            # Track metrics
            epoch_loss += loss_i.item()
            epoch_kl_loss += kl_loss_i.item()
            epoch_recon_loss += recon_loss_i.item()
            tqdm.write(f"Batch {i+1}/{len(dataloader)} - Loss: {loss_i.item():.4f}")
            
            # Step optimizer after accumulating gradients
            optimizer.step()
            scheduler.step()
            
        # Update epoch metrics with batch averages
        epoch_loss /= len(dataloader)
        epoch_kl_loss /=len(dataloader)
        epoch_recon_loss /= len(dataloader)
        
        print(f"Epoch {epoch+1}, Loss: {epoch_loss}")
        run_logger.log({"epoch": epoch+1, "loss": epoch_loss, "recon_loss": epoch_recon_loss, "kl_loss": epoch_kl_loss})

        # Saving Best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            print(f"New best loss: {best_loss}")
            # Save the model
            torch.save(vae.state_dict(), "vae_model.pth")
            try:
                vae.save_pretrained("vae_model")
            except Exception as e:
                print(f"Error saving feature extractor: {e}")
    
    print('Training Complete')


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()

    os.makedirs("results", exist_ok=True)
    # Model Name
    parser.add_argument("--model_name", type=str, default="vae_kl", help="Name of model")

    # Data parameters
    parser.add_argument("--data_id", type=int, default=422, choices=[422, 522], help="Experiment number")
    parser.add_argument("--avg_pooling", type=bool, default=False, help="Apply average pooling to the images")
    
    # Training parameters
    parser.add_argument("--num_epochs", "-e", type=int, default=20, help="Number of epochs for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate training model")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="Weight decay for optimizer")
    parser.add_argument("--beta_recons", type=float, default=0.1, help="weight MSE Loss")
    parser.add_argument("-recons_loss", type=str, default="mse", choices=["mse", "l1"], help="Reconstruction loss type")

    # Pipeline parameters
    parser.add_argument("--data_path", type=str, default=DATA_PATH, help="Path to the data directory")
    parser.add_argument("--batch_size", "-b", type=int, default=4, help="Batch size for training")
 
    args = parser.parse_args()
    run(args)