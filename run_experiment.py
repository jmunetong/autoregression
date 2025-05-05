import os

from accelerate import Accelerator
import wandb
import yaml
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader

from transformers import get_cosine_schedule_with_warmup
from diffusers import AutoencoderKL, VQModel

from utils import get_device, create_directory, print_color
from data_preprocessing import XrdDataset
from losses import IntensityWeightedMSELoss
from trainers import TrainerVAE, TrainerVQ


FEATURE_EXTRACTOR_PATH = "google/vit-base-patch16-224"
DATA_PATH = "data"
URL_MODEL = "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors"

EXPERIMENTS = {
    422: 'mfxl1025422',
    522: 'mfxl1027522'
}

RECONS_LOSS = {
    "mse": nn.MSELoss(reduction="mean"),
    "l1": nn.L1Loss(reduction="mean"),
    "iwmse": IntensityWeightedMSELoss(alpha=2.0),}

MODELS= {"vae_kl": AutoencoderKL,
        "vq": VQModel}

def vae_config_dict(args):
    config = {
        "in_channels": 1,
        "out_channels": 1,
        "latent_channels": args.latent_channels,
        "down_block_types": ("DownEncoderBlock2D",) * 4,
        "up_block_types": ("UpDecoderBlock2D",) * 4,
        "block_out_channels": (64, 128, 256, 512),
        "sample_size": 64,
        "mid_block_add_attention": True
    }
    return config

def vq_config_dict(args):
    config = {
        "in_channels": 1,
        "out_channels": 1,
        "latent_channels": args.latent_channels,
        "down_block_types": ("DownEncoderBlock2D",) * 4,
        "up_block_types": ("UpDecoderBlock2D",) * 4,
        "block_out_channels": (64, 128, 256, 512),
        "sample_size": 64,
        "layers_per_block": 1,
        "act_fn": "silu",
        "sample_size": 32,
        "num_vq_embeddings": 256,
        "norm_num_groups": 32,
    }   
    return config


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
        "latent_channels": args.latent_channels,

    }
    if args.recons_loss == 'iwmse':
        metadata['alpha_mse'] = args.alpha_mse  
    return metadata

def init_configure_model(args):
    """
    Initialize the model configuration based on the provided arguments.
    
    Args:
        args (argparse.Namespace): Command line arguments containing model parameters.
    
    Returns:
        dict: A dictionary containing the model configuration.
    """
    if args.model_name == "vae_kl":
        return vae_config_dict(args), TrainerVAE
    elif args.model_name == "vq":
        return vq_config_dict(args), TrainerVQ
    else:
        raise ValueError(f"Unknown model name: {args.model_name}")


def run(args):
    model_name_dir = args.model_name if not args.test_pipeline else f"{args.model_name}_test"
    model_id, directory = create_directory(model_name_dir, args.data_id)
    experiment_dict = build_experiment_metadata(args)
    torch.cuda.empty_cache()

    # Dataset and Dataloader
    dataset = XrdDataset(data_dir=args.data_path,apply_pooling=args.avg_pooling, data_id=EXPERIMENTS[args.data_id])
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, )
    device = get_device()
    
    print(f'Current CUDA device is:{device}')
    
    # Model Instantiation
    model_config, trainer = init_configure_model(args)
    model = MODELS[args.model_name](**model_config)
    model.train()
    # Optimizer
    num_training_steps = len(dataloader) * args.num_epochs
    num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay= args.weight_decay)

    # Scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps)
    
    # Accelerator instantiation
    accelerator = Accelerator()
    model, optimizer, dataloader, scheduler = accelerator.prepare(
    model, optimizer, dataloader, scheduler)
    model = model.module if hasattr(model, "module") else model
    recons_loss = RECONS_LOSS[args.recons_loss]

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params:,}')
    
    # Loading weights and biases
    run_logger = wandb.init(project=args.model_name, id=model_id, config=args)
    train_pipeline = trainer(args, model, optimizer, scheduler, accelerator, run_logger, recons_loss)
    train_pipeline.run_train(dataloader, experiment_dict, directory)

    with open(os.path.join(directory, "experiment_config.yml"), "w") as f:
        yaml.dump(experiment_dict, f, default_flow_style=False)
    print_color('Training Complete',"green")
    print_color(f"Model information stored in: {directory}", "yellow")


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()

    os.makedirs("results", exist_ok=True)
    # Model Name
    parser.add_argument("--model_name", "-m", type=str, default="vae_kl", help="Name of model")
    parser.add_argument("--latent_channels", type=int, default=2, help="Number of latent channels")

    # Data parameters
    parser.add_argument("--data_id", type=int, default=422, choices=[422, 522], help="Experiment number")
    parser.add_argument("--avg_pooling", type=bool, default=True, help="Apply average pooling to the images")
    
    # Training parameters
    parser.add_argument("--num_epochs", "-e", type=int, default=20, help="Number of epochs for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate training model")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="Weight decay for optimizer")
    parser.add_argument("--beta_recons", type=float, default=1, help="weight MSE Loss")
    parser.add_argument("-recons_loss", "-rls", type=str, default="mse", choices=["mse", "l1", "iwmse"], help="Reconstruction loss type")
    parser.add_argument("--alpha_mse", type=float, default=0.2, help="Alpha for Intensity Weighted MSE Loss")

    # Pipeline parameters
    parser.add_argument("--data_path", type=str, default=DATA_PATH, help="Path to the data directory")
    parser.add_argument("--batch_size", "-b", type=int, default=3, help="Batch size for training")
    parser.add_argument(
        "--test_pipeline", "-t",
        action="store_true",
        help="Enable test pipeline (default: False)"
    )
 
    args = parser.parse_args()
    run(args)