import os 

import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_reconstruction(original: torch.Tensor, reconstructed: torch.Tensor, idx: int, directory: str):
    """
    Plots original and reconstructed images side-by-side.

    Args:
        original (torch.Tensor or np.ndarray): Tensor of shape (B, C, H, W) or (B, H, W)
        reconstructed (torch.Tensor or np.ndarray): Same shape as original
        idx (int): Index in the batch to visualize
    """
    # Convert to numpy if tensors
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu().numpy()
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.detach().cpu().numpy()

    # Handle grayscale or RGB
    if original.ndim == 4:  # (B, C, H, W)
        orig_img = np.transpose(original, (1, 2, 0))
        recon_img = np.transpose(reconstructed, (1, 2, 0))
    elif original.ndim == 3:  # (B, H, W)
        orig_img = original
        recon_img = reconstructed
    else:
        raise ValueError("Expected input shape (B, C, H, W) or (B, H, W)")
    
    # print(f"Original shape: {orig_img.shape}, Reconstructed shape: {recon_img.shape}")  
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(orig_img.squeeze(), cmap='gray' if orig_img.shape[-1] == 1 or orig_img.ndim == 2 else None, vmin = np.percentile(orig_img, 1),
    vmax = np.percentile(orig_img, 99))
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(recon_img.squeeze(), cmap='gray' if recon_img.shape[-1] == 1 or recon_img.ndim == 2 else None, vmin = np.percentile(recon_img, 1),
    vmax = np.percentile(recon_img, 99))
    axes[1].set_title("Reconstructed")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(directory, f'dir{idx}_.png'))