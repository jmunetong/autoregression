import os 

import torch
import numpy as np
import matplotlib.pyplot as plt

def generate_vae_samples(model, dataloader, directory, idx_list):
    count = 0
    n_samples = len(idx_list)
    while count < n_samples:
        batch = next(iter(dataloader))
        for i in range(batch.shape[0]):
            recons = model(batch[i].unsqueeze(0), return_dict=True).sample
            plot_reconstruction(batch[i], recons, idx=idx_list[count], directory=directory)
            count += 1
            del recons
            if count >= n_samples:
                break

def generate_diff_samples(model, diff_model, directory, idx_list, encoding_shape=None, image_shape=None, min_pixel=0, max_pixel=1, use_vae=False):
    for i in idx_list:
        batch = diff_model.sample(batch_size=1)
        plot_fn = plot_output_vae if use_vae else plot_non_vae
        plot_fn(model, batch, i, directory, min_pixel, max_pixel)

def plot_non_vae(model, batch, i, directory, min_pixel, max_pixel):
    min_pixel = np.percentile(transform_to_image(batch), 1)
    max_pixel = np.percentile(transform_to_image(batch), 99)
    plot_diff(batch, directory, idx=i, min_pixel=min_pixel, max_pixel=max_pixel)

def plot_output_vae(model, batch, i, directory, min_pixel, max_pixel):
    out = model.decode(batch.unsqueeze(0), return_dict=True).sample
    min_pixel = np.percentile(transform_to_image(out), 1)
    max_pixel = np.percentile(transform_to_image(out), 99)
    out = out[0]
    if out.dim() == 3:
        out = out.unsqueeze(0)
    plot_diff(out, directory, idx=i, min_pixel=min_pixel, max_pixel=max_pixel)



def transform_to_image(tensor):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    return tensor


def plot_single_image(image: np.ndarray, idx: int, directory: str):
    """
    Plots a single image and saves it to the specified directory.

    Args:
        image (np.ndarray): Image array of shape (H, W) or (H, W, C)
        idx (int): Index for naming the saved file
        directory (str): Directory to save the plot
    """
    plt.imshow(image, cmap='gray' if image.ndim == 2 else None, vmin=np.percentile(image, 1), vmax=np.percentile(image, 99))
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(directory, f'image_{idx}.png'))
    plt.close()

def plot_reconstruction(original: torch.Tensor, reconstructed: torch.Tensor, idx: int, directory: str):
    """
    Plots original and reconstructed images side-by-side.

    Args:
        original (torch.Tensor or np.ndarray): Tensor of shape (B, C, H, W) or (B, H, W)
        reconstructed (torch.Tensor or np.ndarray): Same shape as original
        idx (int): Index in the batch to visualize
    """
    # Convert to numpy if tensors
    original = transform_to_image(original)
    reconstructed = transform_to_image(reconstructed)
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
    
    batch = transform_to_image(batch)
    if batch.ndim ==4:
        assert batch.shape[0] == 1, "Batch should have a single image for plotting"
        batch = batch.squeeze(0)
    

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