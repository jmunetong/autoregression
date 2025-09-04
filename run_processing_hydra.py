import os
import hydra
from omegaconf import DictConfig, OmegaConf
from accelerate import Accelerator
import torch.distributed as dist
import json

import torch
from torch.utils.data import DataLoader

from train_utils.configs import *

from utils import print_color, prepare_state_dict
from plot import generate_vae_samples, generate_diff_samples
from data_preprocessing import create_train_val_datasets_zarr_split


def print_main(accelerator, message, color="white"):
    if accelerator.is_main_process:
        print_color(message, color)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(cfg: DictConfig) -> None:
    # Initialize accelerator
    accelerator = Accelerator(log_with="wandb")
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Convert OmegaConf to a namespace-like object for backward compatibility
    # This allows existing code to access cfg.model_name, cfg.batch_size, etc.
    args = OmegaConf.to_object(cfg)
    
    # Convert to namespace for dot notation access (optional, for compatibility)
    from argparse import Namespace
    args = Namespace(**args)
    
    # Create a shared variable to store the values
    model_id, directory, experiment_dict = prepare_state_dict(args, accelerator)

    # Create datasets on all processes (using same seed ensures consistency)
    train_dataset, val_dataset = create_train_val_datasets_zarr_split(
        data_dir=args.data.data_path,
        data_id=EXPERIMENTS[args.data.data_id],
        train_ratio=args.data.train_ratio,
        random_seed=args.data.seed,
        apply_pooling=args.data.avg_pooling,
        topk=args.data.topk,
    )

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.training.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.training.batch_size, shuffle=False)
    len_dataloader = len(train_dataloader)

    train_dataloader, val_dataloader = accelerator.prepare(train_dataloader, val_dataloader)

    # Convert config to dict for wandb
    args_dict = OmegaConf.to_container(cfg, resolve=True)
    args_dict['model_id'] = model_id
    
    accelerator.init_trackers(
        args.model.model_name,
        config=args_dict
    )
    
    # Validation checks
    if args.model.train_vae_from_checkpoint and args.model.pretrained_vae_path is None:
        raise ValueError("Please provide a path to the pretrained VAE model using pretrained_vae_path")
    
    if args.model.train_diff_from_checkpoint and args.model.pretrained_diff_path is None:
        raise ValueError("Please provide a path to the pretrained diffusion model using pretrained_diff_path")
    
    # VAE/VQ Training
    if not args.diffusion.diff or args.diffusion.latent_diff: 
        print_main(accelerator, f"Running experiment {model_id} with model {args.model.model_name}", "blue")
        
        if args.model.model_name == "vae_kl":
             from train_utils.trainers import TrainerVAE as trainer
        else:
            from train_utils.trainers import TrainerVQ as trainer
            
        trainer_vae = trainer(args, accelerator, len_dataloader)
        
        if args.model.train_vae_from_checkpoint or args.inference.inference:
            loading_directory = args.model.pretrained_vae_path
            print_main(accelerator, f"Loading VAE model from {loading_directory if loading_directory else 'default path'}", "blue")
            trainer_vae.load_model(loading_directory)

        if (args.model.train_vae_from_scratch or args.model.train_vae_from_checkpoint) and not args.inference.inference:
            directory = loading_directory if args.model.pretrained_vae_path else directory
            print_main(accelerator, f"Training VAE model with {len_dataloader} batches", "blue")
            trainer_vae.run_train(train_dataloader, val_dataloader, experiment_dict, directory)
        else:
            print_main(accelerator, "Skipping VAE training. Jumping to inference generation for VAE", "yellow")

        if accelerator.is_main_process:
            model_config = trainer_vae.get_model_config()

    # Latent Diffusion
    if args.diffusion.latent_diff:
        from train_utils.trainers import TrainerDiffusion as train_diff
        vae_model = trainer_vae.get_model()
        diffusion_trainer = train_diff(
            args, 
            vae_model=vae_model, 
            accelerator=accelerator,           
            input_shape=train_dataset.get_image_shape(), 
            len_train_dataloader=len_dataloader
        )
        model_config = diffusion_trainer.get_model_config()

    # Direct Diffusion
    if args.diffusion.diff:
        from train_utils.trainers import TrainerDiffusionNonVAE as train_diff
        print_main(accelerator, f"Running diffusion model without VAE with {len_dataloader} batches", "blue")
        diffusion_trainer = train_diff(
            args, 
            accelerator=accelerator,           
            input_shape=train_dataset.get_image_shape(), 
            len_train_dataloader=len_dataloader
        )
        print_main(accelerator, f"Diffusion model shape: {diffusion_trainer.image_shape}", "blue")

    # Load diffusion checkpoint if needed
    if args.model.train_diff_from_checkpoint or args.inference.inference:
        loading_directory = args.model.pretrained_diff_path
        print_main(accelerator, f"Loading Diffusion model from {loading_directory if loading_directory else 'default path'}", "blue")
        diffusion_trainer.load_model(loading_directory)

    # Train diffusion if needed
    if (args.model.train_diff_from_checkpoint or args.model.train_diff_from_scratch) and not args.inference.inference:
        directory = loading_directory if args.model.pretrained_diff_path else directory
        diffusion_trainer.run_train(train_dataloader, val_dataloader, experiment_dict, directory)

        if accelerator.is_main_process:
            model_config = diffusion_trainer.get_model_config()

    accelerator.wait_for_everyone()

    # Save config
    if accelerator.is_main_process:
        with open(os.path.join(directory, "config.json"), "w") as file:
            json.dump(model_config, file)

        print_color('Training Complete',"green")
        print_color(f"Model information stored in: {directory}", "yellow")
        
    torch.cuda.empty_cache()

    # Sample generation setup
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    per_rank = (args.inference.num_samples + world_size - 1) // world_size
    start = rank * per_rank
    end = min(start + per_rank, args.inference.num_samples)
    accelerator.wait_for_everyone()
    idx_list = list(range(start, end))

    accelerator.end_training()
    
    # Generate samples
    if not args.diffusion.diff and not args.diffusion.latent_diff:
        generate_vae_samples(trainer_vae.get_model(with_accelerator=False).eval(), val_dataloader, directory, idx_list=idx_list)
    else:
        print_main(accelerator, f"Generating {args.inference.num_samples} samples with diffusion model", "blue")
        min_pixel, max_pixel = train_dataset.get_min_max()
        m = None if not args.diffusion.latent_diff else trainer_vae.get_model(with_accelerator=False).eval()
        generate_diff_samples(
            model=m, 
            diff_model=diffusion_trainer.get_model(with_accelerator=False).eval(),  
            directory=directory, 
            idx_list=idx_list,  
            min_pixel=min_pixel, 
            max_pixel=max_pixel, 
            use_vae=args.diffusion.latent_diff
        )
 
        if args.diffusion.latent_diff:
            print_main(accelerator, "Generating samples with VAE", "blue")
            generate_vae_samples(trainer_vae.get_model().eval(), val_dataloader, directory, idx_list=idx_list)


if __name__ == '__main__':
    run()