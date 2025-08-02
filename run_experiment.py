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
from data_preprocessing import XrdDataset


accelerator = Accelerator(log_with="wandb")


def get_args():
    parser = ArgumentParser()

    os.makedirs("results", exist_ok=True)
    # Model Name
    parser.add_argument("--model_name", "-m", type=str, default="vae_kl", help="Name of model")
    parser.add_argument("--latent_channels", type=int, default=4, help="Number of latent channels")
    parser.add_argument("--train_vae", action="store_true", help="Train VAE model")
    parser.add_argument("--inference", action="store_true", help="Run inference on the trained model")

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
    parser.add_argument("--train_from_checkpoint", action="store_true", help="Train from a checkpoint")
    parser.add_argument("--train_from_scratch", action="store_true", help="Train from scratch")
    parser.add_argument("--train_vae_from_checkpoint", action="store_true", help="Train VAE model from a checkpoint")
    parser.add_argument("--train_vae_from_scratch", action="store_true", help="Train VAE model from scratch")


    # Diffusion model arguments
    parser.add_argument("--diff", action="store_true", help="Use diffusion model for training")
    parser.add_argument("--latent_diffisuion", action="store_true", help="Use latent diffusion model")
    parser.add_argument("--use_vae", action='store_true', help="Use VAE model for diffusion training")
    parser.add_argument("--train_diff_from_checkpoint", action="store_true", help="Train diffusion model from a checkpoint")
    parser.add_argument("--train_diff_from_scratch", action="store_true", help="Train diffusion model from scratch")
    parser.add_argument("--pretrained_diff", type=str, default=None, help="Path to pretrained diffusion model")
    parser.add_argument("--pretrained_vae", type=str, default=None, help="Path to pretrained VAE model") #TODO: REMOVE THIS ARGUMENT. Modify it in its correct place

    parser.add_argument("--diff_epochs", type=int, default=10, help="Number of epochs for diffusion model training")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size for diffusion model")
    parser.add_argument("--vit_size", type=str, default="base", choices=["base", "large", "huge"], help="Size of the VIT model")
    parser.add_arugment("--patch_size", type=int, default=16, help="Patch size for the VIT model")

    # Inference parameters #TODO: Complete this part for running inference values
    parser.add_argument("--inference", action="store_true", help="Run inference on the trained model")
    parser.add_argument("--generate_samples", "-gs", action="store_true", help="Generate samples after training")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate")
    
    args = parser.parse_args()
    return args


def run(args):   
    # Create a shared variable to store the values
    model_id, directory, experiment_dict = prepare_state_dict(args, accelerator)
    # Dataset and Dataloader
    dataset = XrdDataset(data_dir=args.data_path,apply_pooling=args.avg_pooling, data_id=EXPERIMENTS[args.data_id], top_k=args.topk)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    len_dataloader = len(dataloader)
    dataloader = accelerator.prepare(dataloader)
    
    args_dict = vars(args)
    args_dict['model_id'] = model_id
    accelerator.init_trackers(
        args.model_name,
        config=args_dict
    )
    
    if not args.diff or args.latent_diffisuion: 
        # Train a VAE or VQ model either for generative modeling or to train A vae model for latent diffusion.
        print_color(f"Training {args.model_name} model", "blue")
        if args.model_name == "vae_kl":
             from train_utils.trainers import TrainerVAE as trainer
        else:
            from train_utils.trainers import TrainerVQ as trainer
        trainer_vae = trainer(args, accelerator,len_dataloader)
    
    
        if args.train_vae_from_checkpoint or args.inference: #TODO: FIX THE INFERENCE PARAMETER 
            loading_directory = args.pretrained_vae
            print_color(f"Loaded VAE model from {loading_directory if loading_directory else 'default path'}", "blue")
            trainer_vae.load_model(loading_directory)

        if args.train_vae_from_scratch or args.train_vae_from_checkpoint:
            if accelerator.is_main_process:
                print_color("Training VAE model", "blue")

        else:
            if accelerator.is_main_process:
                print_color("Skipping VAE training", "yellow")
                print_color("Jumping to inference generation for VAE model", "yellow")
          
            # logging.info(f"Loaded model from {loading_directory if loading_directory else 'default path'}")
        if accelerator.is_main_process:
            model_config = trainer_vae.get_model_config()


    if args.latent_diff:
        from train_utils.trainers import TrainerDiffusion as train_diff
        vae_model = trainer_vae.get_model(with_accelerator=False)
        
        diffusion_trainer = train_diff(args, vae_model=vae_model, accelerator=accelerator,           input_shape=dataset.get_image_shape(), len_train_data_loader=len(dataloader))
        model_config = diffusion_trainer.get_model_config()
        

    if args.diff:
        from train_utils.trainers import TrainerDiffusionNonVAE as train_diff
        if accelerator.is_main_process:
            print_color("Training Diffusion model without VAE", "blue")
        diffusion_trainer = train_diff(args, accelerator=accelerator,           input_shape=dataset.get_image_shape(), len_train_data_loader=len(dataloader))

        if accelerator.is_main_process:
            print_color(f"Diffusion model shape: {diffusion_trainer.image_shape}", "blue")

    if args.train_diff_from_checkpoint or args.inference:
        loading_directory = args.pretrained_diff #TODO: ADD THIS ARGUMENT
        if accelerator.is_main_process:
            print_color(f"Loaded Diffusion model from {loading_directory if loading_directory else 'default path'}", "blue")
    
    if args.train_diff_from_checkpoint or args.train_diff_from_scratch:
        #todo: directory needs to change 
        directory = loading_directory if not args.train_diff_from_scratch else directory
        diffusion_trainer.run_train(dataloader, experiment_dict, directory)
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
    if not args.inference or not args.generate_samples:
        print_color("Skipping inference generation", "yellow")
        return
    
    if not args.diff or not args.latent_diff:
        generate_vae_samples(trainer_vae.get_model().eval(), dataloader, directory)
    else:
        #TODO: MAKE THE INFERENCE GENERATION BE ACROSS EACH GPU.
        samples = args.num_samples if args.num_samples > 0 else 10
        if accelerator.is_main_process:
            print_color(f"Generating {samples} samples with diffusion model", "blue")
        min_pixel, max_pixel = dataset.get_min_max()
        #TODO: MODIFY THIS FUNCTION FOR DIFFUSION WITHOUT VAE BACKEND.

        generate_diff_samples(diffusion_trainer.get_model().eval(),  directory, idx_list=idx_list, encoding_shape=diffusion_trainer.encoding_shape, image_shape=diffusion_trainer.image_shape, min_pixel=min_pixel, max_pixel=max_pixel, use_vae=args.use_vae)

        if args.use_vae:
            print_color("Generating samples with VAE", "blue")
            generate_vae_samples(trainer_vae.get_model().eval(), dataloader, directory, idx_list=idx_list)
        # generate_vae_samples(trainer_vae.get_model(), dataloader, directory)


    


if __name__ == '__main__':
    args = get_args()
    run(args)
    # try:
        
    # except Exception as e:
    #     accelerator.end_training()
    #     if accelerator.is_main_process:
    #         print_color("Experiment Failed", "red")
    #         print(f"❌ Failed to compute: {e}")

    # elif args.use_vae: #TODO: There might be a better logic to handle this case

    #     # This is for training a diffusion model with a VAE backend. This assumes that the VAE model has been trained and is ready to be used.
    #     if args.model_name == "vae_kl":
    #          from train_utils.trainers import TrainerVAE as trainer
    #     else:
    #         from train_utils.trainers import TrainerVQ as trainer
    #     train_pipeline = trainer(args, accelerator,len_dataloader)
    #     train_pipeline.load_model() # TODO: Pass the correct parameters for the model
    #     model = train_pipeline.get_model(with_accelerator=False)
    #     model = accelerator.prepare(model)
    #     accelerator.wait_for_everyone()
    #     if accelerator.is_main_process:
    #         print_color(f"Loaded {args.model_name} model for diffusion training", "blue")


        # with open(os.path.join(args.pretrained_vae, "config.json"), "r") as f:
        #     model_config_load = json.load(f)
        # accelerator.wait_for_everyone()
        
        # safe_tensor_path = os.path.join(args.pretrained_vae, "diffusion_pytorch_model.safetensors")
   
        # if not os.path.exists(safe_tensor_path):
        #     if accelerator.is_main_process:
        #         print_color(f"Model file not found at {safe_tensor_path}. Please check the path.", "red")
        #     return
        # else:
        #     model = AutoencoderKL.from_pretrained(
        #         args.pretrained_vae,
        #     )

        #     model = accelerator.prepare(model)
        #     accelerator.wait_for_everyone() #TODO: use the trainer to load the parameters and do not do it here