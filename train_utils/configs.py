from diffusers import AutoencoderKL, VQModel
import torch.nn as nn

from train_utils.losses import IntensityWeightedMSELoss
from train_utils.trainers import TrainerVAE, TrainerVQ
from models.diff.autoregressive_diffusion import ImageAutoregressiveDiffusion as diff_model

FEATURE_EXTRACTOR_PATH = "google/vit-base-patch16-224"
DATA_PATH = "data"
URL_MODEL = "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors"

EXPERIMENTS = {
    422: 'mfxl1025422',
    522: 'mfxl1027522'
}

VIT_MODELS = {"base": dict(
                dim = 768,
                depth = 12,
                heads =12),
            "large": dict(
                dim= 1024,
                depth=24, 
                heads=16),
            "huge": dict(
                dim= 1280,
                depth=32,
                heads=20)}

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
        "block_out_channels": (32, 64, 128, 128),
        "sample_size": 64,
        "mid_block_add_attention": True
    }
    return config

def vq_config_dict(args):
    config = {
        "in_channels": 1,
        "out_channels": 1,
        "latent_channels": args.latent_channels,
        "down_block_types":("DownEncoderBlock2D",) * 4,
        "up_block_types": ("UpDecoderBlock2D",) * 4,
        "block_out_channels": (32, 64, 128, 128),
        "sample_size": 64,
        "layers_per_block": 1,
        "act_fn": "silu",
        "sample_size": 32, #TODO: Modify this to the actual image size
        "num_vq_embeddings": 256,
        "norm_num_groups": 32,
        "scaling_factor": 1,
        "norm_type": "spatial"
    }   
    return config

def init_configure_model(args):
    """
    Initialize the model configuration based on the provided arguments.
    
    Args:
        args (argparse.Namespace): Command line arguments containing model parameters.
    
    Returns:
        dict: A dictionary containing the model configuration.
    """
    if args.model_name == "vae_kl":
        return vae_config_dict(args)
    elif args.model_name == "vq":
        return vq_config_dict(args)
    else:
        raise ValueError(f"Unknown model name: {args.model_name}")


def init_configure_vit(vit_size, patch_size, input_shape):
    model_dim = VIT_MODELS[vit_size]
    return diff_model(model=model_dim, image_size=input_shape[-1], patch_size=patch_size)