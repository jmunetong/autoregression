import os
import numpy as np
import torch

from torch.utils.data import Dataset
from utils import get_directories, get_imgs, get_device


path = 'data'
directories = get_directories(path)[:1]
imgs,_ = get_imgs(directories) # shape (40, 1920, 1920, 1)

def preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    Preprocess a grayscale image for use with a VAE model:
    1. Center crop to (1792, 1792)
    2. Mask edges (10-pixel border) to 0 before pooling
    3. Average pool to (256, 256)
    4. Convert to 3-channel image in (3, 256, 256) format

    Args:
        img (np.ndarray): Input grayscale image of shape (1920, 1920, 1)

    Returns:
        np.ndarray: Preprocessed image of shape (3, 256, 256)
    """
    if img.shape != (1920, 1920, 1):
        raise ValueError(f"Expected input shape (1920, 1920, 1), got {img.shape}")

    img_2d = img.squeeze()  # (1920, 1920)

    # Step 1: Center crop to (1792, 1792)
    pool_size = 1920 // 256  # 7
    trimmed_size = pool_size * 256  # 1792
    start = (1920 - trimmed_size) // 2
    end = start + trimmed_size
    img_cropped = img_2d[start:end, start:end]  # shape: (1792, 1792)

    # Step 2: Apply edge mask (pre-pooling)
    mask = np.ones_like(img_cropped, dtype=img_cropped.dtype)
    mask[:10, :] = 0
    mask[-10:, :] = 0
    mask[:, :10] = 0
    mask[:, -10:] = 0
    img_masked = img_cropped * mask

    # Step 3: Average pooling to (256, 256)
    img_reshaped = img_masked.reshape(256, pool_size, 256, pool_size)
    img_pooled = img_reshaped.mean(axis=(1, 3))  # shape: (256, 256)

    # Step 4: Repeat channel and convert to channel-first (3, 256, 256)
    img_pooled = img_pooled[:, :, np.newaxis]  # (256, 256, 1)
    img_rgb = np.repeat(img_pooled, 3, axis=2)  # (256, 256, 3)
    img_final = np.transpose(img_rgb, (2, 0, 1))  # (3, 256, 256)

    return img_final



class XrdDataset(Dataset):
    def __init__(self, data_dir, get_directories, get_imgs, feature_extractor=None, img_blocks=40):
        """
        Args:
            data_dir (str): Path to folder containing .zarr data files.
            get_directories (callable): Function that returns list of zarr paths inside `data_dir`.
            get_imgs (callable): Function that loads images from given zarr paths.
            img_blocks (int): Number of images per zarr file (default 40).
        """
        self.image_dir = data_dir
        self.image_files = get_directories(data_dir)  # e.g., ["a.zarr", "b.zarr", ...]
        self.img_blocks = img_blocks

        # Create flat indexing: total number of images = num_files × images_per_file
        self.total_images = len(self.image_files) * img_blocks

    def __len__(self):
        return self.total_images

    def __getitem__(self, index):
        # Determine which file and which image within it
        file_idx = index // self.img_blocks
        inner_idx = index % self.img_blocks

        # Load only the necessary zarr file
        zarr_path = self.image_files[file_idx]
        print(f"Loading {zarr_path}")

        images, _ = get_imgs([zarr_path])  # should return shape (40, 3, 256, 256)

        img = images[inner_idx]  # select single image
        return torch.from_numpy(img).float()  # ensure torch.Tensor [3, 256, 256]
        

