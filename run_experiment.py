import os
from argparse import ArgumentParser

from accelerate import Accelerator
import torch.distributed as dist
import json

import torch
from torch.utils.data import DataLoader

from train_utils.configs import *

from utils import print_color, prepare_state_dict
from plot import generate_vae_samples, generate_diff_samples
from data_preprocessing import create_train_val_datasets_zarr_split


accelerator = Accelerator(log_with="wandb")


def print_main(accelerator, message, color="white"):
    if accelerator.is_main_process:
        print_color(message, color)


def get_args():
    parser = ArgumentParser()
    os.makedirs("results", exist_ok=True)
    
    # Training parameters
    parser.add_argument("--model_name", "-m", type=str, default="vae_kl", help="Name of model")
    parser.add_argument("--batch_size", "-b", type=int, default=3, help="Batch size for training")
    parser.add_argument(
        "--test_pipeline", "-t",
        action="store_true",
        help="Enable test pipeline (default: False)"
    )
    parser.add_argument("--num_epochs", "-e", type=int, default=20, help="Number of epochs for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate training model")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="Weight decay for optimizer")
    parser.add_argument("--beta_recons", type=float, default=0.5, help="weight MSE Loss")
    parser.add_argument("--recons_loss", "-rls", type=str, default="mse", choices=["mse", "l1", "iwmse"], help="Reconstruction loss type")
    parser.add_argument("--alpha_mse", type=float, default=2.0, help="Alpha value for Intensity Weighted MSE Loss")
    parser.add_argument("--ema_decay", type=float, default=0.9999, help="EMA decay rate")

    # Inference parameters #TODO: Complete this part for running inference values
    parser.add_argument("--inference", action="store_true", help="Run inference on the trained model")
    parser.add_argument("--generate_samples", "-gs", action="store_true", help="Generate samples after training")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate")

    # Data parameters
    parser.add_argument("--data_id", type=int, default=522, choices=[422, 522], help="Experiment number")
    parser.add_argument("--avg_pooling", action='store_true', help="Apply average pooling to the images")
    parser.add_argument("--topk", type=float, default=1.0, help="Top k percent of images to use for training")
    parser.add_argument("--data_path", type=str, default=DATA_PATH, help="Path to the data directory")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of training data to validation data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    # VQ and VAE model parameters
    parser.add_argument("--latent_channels", type=int, default=4, help="Number of latent channels")
    parser.add_argument("--use_annealing", "-ua", action="store_true", help="Use annealing for KL divergence loss")
    parser.add_argument("--annealing_shape", type=str, default="cosine", choices=["linear", "cosine", "logistic"], help="Shape of the annealing function")
    parser.add_argument("--train_vae_from_checkpoint", action="store_true", help="Train VAE model from a checkpoint")
    parser.add_argument("--train_vae_from_scratch", action="store_true", help="Train VAE model from scratch")
    parser.add_argument("--pretrained_vae_path", type=str, default=None, help="Path to pretrained VAE model")

    # Diffusion model arguments
    parser.add_argument("--diff", action="store_true", help="Use diffusion model for training")
    parser.add_argument("--latent_diff", action="store_true", help="Use latent diffusion model")
    parser.add_argument("--train_diff_from_checkpoint", action="store_true", help="Train diffusion model from a checkpoint")
    parser.add_argument("--train_diff_from_scratch", action="store_true", help="Train diffusion model from scratch")
    parser.add_argument("--pretrained_diff_path", type=str, default=None, help="Path to pretrained diffusion model")
    parser.add_argument("--diff_epochs", type=int, default=10, help="Number of epochs for diffusion model training")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size for diffusion model")
    parser.add_argument("--vit_size", type=str, default="base", choices=["base", "large", "huge"], help="Size of the VIT model")

    
    args = parser.parse_args()
    return args


def run(args):   
    # Create a shared variable to store the values
    model_id, directory, experiment_dict = prepare_state_dict(args, accelerator)

    # Create datasets on all processes (using same seed ensures consistency)
    train_dataset, val_dataset = create_train_val_datasets_zarr_split(
        data_dir=args.data_path,
        data_id=EXPERIMENTS[args.data_id],
        train_ratio=args.train_ratio,
        random_seed=args.seed,
        apply_pooling=args.avg_pooling,
        topk=args.topk,
    )

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    len_dataloader = len(train_dataloader)

    train_dataloader, val_dataloader = accelerator.prepare(train_dataloader, val_dataloader)

    args_dict = vars(args)
    args_dict['model_id'] = model_id
    accelerator.init_trackers(
        args.model_name,
        config=args_dict
    )
    if args.train_vae_from_checkpoint and args.pretrained_vae_path is None:
        raise ValueError("Please provide a path to the pretrained VAE model using --pretrained_vae_path")
    
    if args.train_diff_from_checkpoint and args.pretrained_diff_path is None:
        raise ValueError("Please provide a path to the pretrained diffusion model using --pretrained_diff")
    
    
    if not args.diff or args.latent_diff: 
        # Train a VAE or VQ model either for generative modeling or to train A vae model for latent diffusion.
        print_main(accelerator, f"Running experiment {model_id} with model {args.model_name}", "blue")
        if args.model_name == "vae_kl":
             from train_utils.trainers import TrainerVAE as trainer
        else:
            from train_utils.trainers import TrainerVQ as trainer
        trainer_vae = trainer(args, accelerator,len_dataloader)
        if args.train_vae_from_checkpoint or args.inference:
            loading_directory = args.pretrained_vae_path
            print_main(accelerator, f"Loading VAE model from {loading_directory if loading_directory else 'default path'}", "blue")
            trainer_vae.load_model(loading_directory)

        if (args.train_vae_from_scratch or args.train_vae_from_checkpoint) and not args.inference:
            directory = loading_directory if args.pretrained_vae_path else directory
            print_main(accelerator, f"Training VAE model with {len_dataloader} batches", "blue")
            trainer_vae.run_train(train_dataloader, val_dataloader, experiment_dict, directory)

        else:
            print_main(accelerator, "Skipping VAE training. Jumping to inference generation for VAE", "yellow")

        if accelerator.is_main_process:
            model_config = trainer_vae.get_model_config()


    if args.latent_diff:
        from train_utils.trainers import TrainerDiffusion as train_diff
        vae_model = trainer_vae.get_model()
        diffusion_trainer = train_diff(args, vae_model=vae_model, accelerator=accelerator,           input_shape=train_dataset.get_image_shape(), len_train_dataloader=len_dataloader)
        model_config = diffusion_trainer.get_model_config()
        

    if args.diff:
        from train_utils.trainers import TrainerDiffusionNonVAE as train_diff
        print_main(accelerator, f"Running diffusion model without VAE with {len_dataloader} batches", "blue")
        diffusion_trainer = train_diff(args, accelerator=accelerator,           input_shape=train_dataset.get_image_shape(), len_train_dataloader=len_dataloader)
        print_main(accelerator, f"Diffusion model shape: {diffusion_trainer.image_shape}", "blue")

    if args.train_diff_from_checkpoint or args.inference:
        loading_directory = args.pretrained_diff_path #TODO: ADD THIS ARGUMENT
        print_main(accelerator, f"Loading Diffusion model from {loading_directory if loading_directory else 'default path'}", "blue")
        diffusion_trainer.load_model(loading_directory)

    if (args.train_diff_from_checkpoint or args.train_diff_from_scratch) and not args.inference:
        #todo: directory needs to change
        directory = loading_directory if args.pretrained_diff_path else directory
        diffusion_trainer.run_train(train_dataloader,val_dataloader, experiment_dict, directory)

        if accelerator.is_main_process:
            model_config = diffusion_trainer.get_model_config()

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        with open(os.path.join(directory, "config.json"), "w") as file:
            json.dump(model_config, file)

        print_color('Training Complete',"green")
        print_color(f"Model information stored in: {directory}", "yellow")
        
        
    torch.cuda.empty_cache()

    #### This is to make sure we have a given number of samples per GPU
    rank = accelerator.process_index # gpu rank
    world_size = accelerator.num_processes # total number of gpus
    per_rank = (args.num_samples + world_size - 1) // world_size # number of samples per gpu
    start = rank * per_rank
    end = min(start + per_rank, args.num_samples)
    accelerator.wait_for_everyone()
    idx_list = list(range(start, end))

    accelerator.end_training()
    # Inference generation
    if not args.diff and not args.latent_diff:
        generate_vae_samples(trainer_vae.get_model(with_accelerator=False).eval(), val_dataloader, directory, idx_list=idx_list)
    else:
        print_main(accelerator, f"Generating {args.num_samples} samples with diffusion model", "blue")
        min_pixel, max_pixel = train_dataset.get_min_max()
        m=None if not args.latent_diff else trainer_vae.get_model(with_accelerator=False).eval()
        generate_diff_samples(model=m, diff_model=diffusion_trainer.get_model(with_accelerator=False).eval(),  directory=directory, idx_list=idx_list,  min_pixel=min_pixel, max_pixel=max_pixel, use_vae=args.latent_diff)
 
        if args.latent_diff:
            print_main(accelerator, "Generating samples with VAE", "blue")
            generate_vae_samples(trainer_vae.get_model().eval(), val_dataloader, directory, idx_list=idx_list)




if __name__ == '__main__':
    args = get_args()
    run(args)
  