import os
# CRITICAL: Set these BEFORE any torch/accelerate imports
os.environ['TORCH_DISTRIBUTED_BACKEND'] = 'gloo'
os.environ['NCCL_DISABLED'] = '1'
os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '0'

import json
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig

from accelerate import Accelerator
import torch.distributed as dist
import torch
from torch.utils.data import DataLoader

from train_utils.configs import *
from utils import print_color, prepare_state_dict, create_args_compatibility
from plot import generate_vae_samples, generate_diff_samples
from data_preprocessing import create_train_val_datasets_zarr_split



def print_main(accelerator, message, color="white"):
    if accelerator.is_main_process:
        print_color(message, color)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(cfg: DictConfig) -> None:
    # Ensure results directory exists
    accelerator = Accelerator(log_with="wandb")
    
    # Create compatibility layer - convert Hydra config to args-like object
    args = create_args_compatibility(cfg)
    
    out_dir = HydraConfig.get().runtime.output_dir
 
    # Create a shared variable to store the values (pass args, not cfg!)
    directory, experiment_dict = prepare_state_dict(args, accelerator, out_dir)

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

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    accelerator.init_trackers(
        args.model_name,
        config=cfg_dict,
        init_kwargs={
            "wandb": {
                "dir": out_dir, 
            }
        },
    )
        
    if args.train_vae_from_checkpoint and args.pretrained_vae_path is None:
        raise ValueError("Please provide a path to the pretrained VAE model using pretrained_vae_path")
    
    if args.train_diff_from_checkpoint and args.pretrained_diff_path is None:
        raise ValueError("Please provide a path to the pretrained diffusion model using pretrained_diff_path")
    
    
    # Use original logic with args object
    if not args.diff or args.latent_diff: 
        # Train a VAE or VQ model either for generative modeling or to train A vae model for latent diffusion.
        print_main(accelerator, f"Running experiment {args.model_name}", "blue")
        if args.model_name == "vae_kl":
             from train_utils.trainers import TrainerVAE as trainer
        else:
            from train_utils.trainers import TrainerVQ as trainer
        trainer_vae = trainer(args, accelerator, len_dataloader)
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
        diffusion_trainer = train_diff(args, vae_model=vae_model, accelerator=accelerator, 
                                     input_shape=train_dataset.get_image_shape(), len_train_dataloader=len_dataloader)
        model_config = diffusion_trainer.get_model_config()
        

    if args.diff:
        from train_utils.trainers import TrainerDiffusionNonVAE as train_diff
        print_main(accelerator, f"Running diffusion model without VAE with {len_dataloader} batches", "blue")
        diffusion_trainer = train_diff(args, accelerator=accelerator, 
                                     input_shape=train_dataset.get_image_shape(), len_train_dataloader=len_dataloader)
        print_main(accelerator, f"Diffusion model shape: {diffusion_trainer.image_shape}", "blue")

        if args.train_diff_from_checkpoint or args.inference:
            loading_directory = args.pretrained_diff_path
            print_main(accelerator, f"Loading Diffusion model from {loading_directory if loading_directory else 'default path'}", "blue")
            diffusion_trainer.load_model(loading_directory)

        if (args.train_diff_from_checkpoint or args.train_diff_from_scratch) and not args.inference:
            directory = loading_directory if args.pretrained_diff_path else directory
            diffusion_trainer.run_train(train_dataloader, val_dataloader, experiment_dict, directory)

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
    print("Running Experiment")
    run()