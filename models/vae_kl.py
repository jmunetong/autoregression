import torch
from diffusers import AutoencoderKL

def build_vae_model(
    in_channels=3,
    out_channels=3,
    latent_channels=4,
    down_block_types=("DownEncoderBlock2D",) * 4,
    up_block_types=("UpDecoderBlock2D",) * 4,
    block_out_channels=(64, 128, 256, 512),
    sample_size=64
):
    """
    Build a VAE model with the specified configuration.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        latent_channels (int): Number of latent channels.
        down_block_types (tuple): Types of downsampling blocks.
        up_block_types (tuple): Types of upsampling blocks.
        block_out_channels (tuple): Number of output channels for each block.
        sample_size (int): Size of the input samples.

    Returns:
        AutoencoderKL: Configured VAE model.
    """
    
    vae = AutoencoderKL(
        in_channels=in_channels,
        out_channels=out_channels,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        block_out_channels=block_out_channels,
        latent_channels=latent_channels,
        sample_size=sample_size
    )
    
    return vae