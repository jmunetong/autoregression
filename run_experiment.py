import os
from argparse import ArgumentParser

from accelerate import Accelerator, load_checkpoint_in_model, infer_auto_device_map
import wandb
import torch.distributed as dist
import json

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
from transformers import get_cosine_schedule_with_warmup
from icecream import ic

from configs import *

from utils import print_color, prepare_state_dict, generate_vae_samples, generate_diff_samples
from data_preprocessing import XrdDataset
from train_utils.trainers import TrainerVAE, TrainerVQ, TrainerDiffusion, TrainerDiffusionNonVAE

accelerator = Accelerator(log_with="wandb")


def get_args():
    parser = ArgumentParser()

    os.makedirs("results", exist_ok=True)
    # Model Name
    parser.add_argument("--model_name", "-m", type=str, default="vae_kl", help="Name of model")
    parser.add_argument("--latent_channels", type=int, default=4, help="Number of latent channels")

    # Data parameters
    parser.add_argument("--data_id", type=int, default=522, choices=[422, 522], help="Experiment number")
    parser.add_argument("--avg_pooling", action='store_true', help="Apply average pooling to the images")
    parser.add_argument("--topk", type=float, default=1.0, help="Top k percent of images to use for training")

    # Training parameters
    parser.add_argument("--num_epochs", "-e", type=int, default=20, help="Number of epochs for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate training model")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="Weight decay for optimizer")
    parser.add_argument("--beta_recons", type=float, default=0.5, help="weight MSE Loss")
    parser.add_argument("-recons_loss", "-rls", type=str, default="mse", choices=["mse", "l1", "iwmse"], help="Reconstruction loss type")
    parser.add_argument("--alpha_mse", type=float, default=2.0, help="Alpha value for Intensity Weighted MSE Loss")

    # Pipeline parameters
    parser.add_argument("--data_path", type=str, default=DATA_PATH, help="Path to the data directory")
    parser.add_argument("--batch_size", "-b", type=int, default=3, help="Batch size for training")
    parser.add_argument(
        "--test_pipeline", "-t",
        action="store_true",
        help="Enable test pipeline (default: False)"
    )

    # Arguments for variational autoencoder.
    parser.add_argument("--use_annealing", "-ua", action="store_true", help="Use annealing for KL divergence loss")
    parser.add_argument("--annealing_shape", type=str, default="cosine", choices=["linear", "cosine", "logistic"], help="Shape of the annealing function")
    
    # Diffusion model arguments
    parser.add_argument("--diff", action="store_true", help="Use diffusion model for training")
    parser.add_argument("--use_vae", action='store_true', help="Use VAE model for diffusion training")
    parser.add_argument("--train_vae", action="store_true", help="Train VAE model")
    parser.add_argument("--pretrained_vae", type=str, default=None, help="Path to pretrained VAE model")
    parser.add_argument("--diff_epochs", type=int, default=10, help="Number of epochs for diffusion model training")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size for diffusion model")
    parser.add_argument("--vit_size", type=str, default="base", choices=["base", "large", "huge"], help="Size of the VIT model")
    
    args = parser.parse_args()
    return args


def run(args):   
    # Create a shared variable to store the values
    model_id, directory, experiment_dict = prepare_state_dict(args, accelerator)
    # Dataset and Dataloader
    dataset = XrdDataset(data_dir=args.data_path,apply_pooling=args.avg_pooling, data_id=EXPERIMENTS[args.data_id], top_k=args.topk)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, )

    # Model Instantiation

    
    
    # Configuring Optimizer steps
    #########################################
    # TODO: DO THIS INSIDE TRAINER
    model_config, trainer = init_configure_model(args)
    model = MODELS[args.model_name](**model_config)
    model.train()
    num_training_steps = len(dataloader) * args.num_epochs
    num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay= args.weight_decay)

    # Scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps)
    
    # Accelerator instantiation
    model, optimizer, dataloader, scheduler = accelerator.prepare(
    model, optimizer, dataloader, scheduler)

    model = model.module if hasattr(model, "module") else model
    recons_loss = RECONS_LOSS[args.recons_loss]
    ## ALL THIS INSIDE THE GIVEN MODEL 
    #################################

    dataloader = accelerator.prepare(dataloader)
    if accelerator.is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        print(f'Total parameters: {total_params:,}')
        
    args_dict = vars(args)
    args_dict['model_id'] = model_id
    accelerator.init_trackers(
        args.model_name,
        config=args_dict
    )
    
    if not args.diff or (args.diff and args.train_vae):
        print_color("Training VAE model", "blue")
        train_pipeline = trainer(args, model, optimizer, scheduler, accelerator, recons_loss)
        train_pipeline.run_train(dataloader, experiment_dict, directory)
    elif args.use_vae:
        with open(os.path.join(args.pretrained_vae, "config.json"), "r") as f:
            model_config_load = json.load(f)
        accelerator.wait_for_everyone()
        
        safe_tensor_path = os.path.join(args.pretrained_vae, "diffusion_pytorch_model.safetensors")
   
        if not os.path.exists(safe_tensor_path):
            if accelerator.is_main_process:
                print_color(f"Model file not found at {safe_tensor_path}. Please check the path.", "red")
            return
        else:
            model = AutoencoderKL.from_pretrained(
                args.pretrained_vae,
            )

            model = accelerator.prepare(model)
            accelerator.wait_for_everyone()

    else:
        pass
    if args.diff:
        diff_model = init_configure_vit(args.vit_size, args.patch_size, dataset.get_image_shape())
        if args.use_vae:
            if accelerator.is_main_process:
                print_color("Training Diffusion model with VAE", "blue")
            diffusion_trainer = TrainerDiffusion(args, model, diff_model, scheduler, accelerator, image_shape = dataset.get_image_shape(),learning_rate=args.lr, patch_size=args.patch_size)
        else:
            del model
            if accelerator.is_main_process:
                print_color("Training Diffusion model without VAE", "blue")
            diffusion_trainer = TrainerDiffusionNonVAE(args, diff_model, scheduler, accelerator, patch_size=args.patch_size, image_shape = dataset.get_image_shape(), learning_rate=args.lr, len_train_data_loader=len(dataloader), num_epochs=args.diff_epochs)
        if accelerator.is_main_process:
            print_color(f"Diffusion model shape: {diffusion_trainer.image_shape}", "blue")
       
        diffusion_trainer.run_train(dataloader, experiment_dict, directory)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        with open(os.path.join(directory, "config.json"), "w") as file:
            json.dump(model_config, file)

        print_color('Training Complete',"green")
        print_color(f"Model information stored in: {directory}", "yellow")
        model.eval()
        
        torch.cuda.empty_cache()
        if not args.diff:
            generate_vae_samples(model, dataloader, directory)
        else:
            #TODO: MAKE THE INFERENCE GENERATION BE ACROSS EACH GPU.
            samples = 10
            min_pixel, max_pixel = dataset.get_min_max()
            #TODO: MODIFY THIS FUNCTION FOR DIFFUSION WITHOUT VAE BACKEND.
            generate_diff_samples(diffusion_trainer.unwrap(model), diffusion_trainer.get_diff_model(), directory,samples, diffusion_trainer.encoding_shape, diffusion_trainer.image_shape, min_pixel, max_pixel, args.use_vae)

            if args.use_vae:
                print_color("Generating samples with VAE", "blue")
                generate_vae_samples(model, dataloader, directory)
            generate_vae_samples(diffusion_trainer.unwrap(model), dataloader, directory)

    accelerator.end_training()
    


if __name__ == '__main__':
    args = get_args()
    run(args)
    # try:
        
    # except Exception as e:
    #     accelerator.end_training()
    #     if accelerator.is_main_process:
    #         print_color("Experiment Failed", "red")
    #         print(f"❌ Failed to compute: {e}")