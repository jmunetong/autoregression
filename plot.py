import os 

import torch
import numpy as np
import matplotlib.pyplot as plt

def trainsform_to_image(tensor):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    return tensor


def plot_reconstruction(original: torch.Tensor, reconstructed: torch.Tensor, idx: int, directory: str):
    """
    Plots original and reconstructed images side-by-side.

    Args:
        original (torch.Tensor or np.ndarray): Tensor of shape (B, C, H, W) or (B, H, W)
        reconstructed (torch.Tensor or np.ndarray): Same shape as original
        idx (int): Index in the batch to visualize
    """
    # Convert to numpy if tensors
    original = trainsform_to_image(original)
    reconstructed = trainsform_to_image(reconstructed)
    # Handle grayscale or RGB
    if original.ndim == 4:  # (B, C, H, W)
        original = np.transpose(original, (1, 2, 0))
        reconstructed = np.transpose(reconstructed, (1, 2, 0))

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(original.squeeze(), cmap='gray' if original.shape[-1] == 1 or original.ndim == 2 else None, vmin = np.percentile(original, 1),
    vmax = np.percentile(original, 99))
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(reconstructed.squeeze(), cmap='gray' if reconstructed.shape[-1] == 1 or reconstructed.ndim == 2 else None, vmin = np.percentile(reconstructed, 1),
    vmax = np.percentile(reconstructed, 99))
    axes[1].set_title("Reconstructed")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(directory, f'dir{idx}_.png'))

def plot_diff(batch, directory, idx=0, min_pixel=None, max_pixel=None):
    
    batch = trainsform_to_image(batch)

    batch = np.transpose(batch, (1, 2, 0))
    if min_pixel is None:
        min_pixel = np.percentile(batch, 1)
    if max_pixel is None:
        max_pixel = np.percentile(batch, 99)

    plt.imshow(batch, vmin=min_pixel, vmax = max_pixel, cmap='viridis' if batch.shape[-1] == 1 else None)
    plt.axis("off")
    plt.tight_layout()
    plt.title("Diffusion Process Reconstruction")
    plt.savefig(os.path.join(directory, f'diff_dir{idx}_.png'))
    plt.close()