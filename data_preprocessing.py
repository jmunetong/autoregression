
import os
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

from utils import get_directories, get_imgs

# # Data preprocessing using standard normalization
# def preprocess_images(img:np.ndarray):
#     mean_value = img.mean()
#     img = (img - mean_value) / (img.std() + 1e-6)
#     if img.ndim == 3:
#         img = img.expand_dims(0)
#     if img.shape[-1]==1:
#         img = img.repeat(3, axis=-1)
#     img[:,10:, :] = 0
#     img[:,:, :10]  = 0
#     img[:,-10:,:]  = 0
#     img[:,:, -10:] = 0
#     print("min", img.min(), "max",img.max())
#     return img

# Data preprocessing using min-max normalization
def preprocess_images(img:np.ndarray):
    # img = np.log1p(img)
    print("min before normalization", img.min(), "max before normalization",img.max())
    min_value = img.min()
    max_value = img.max()
    img = (img - min_value) / (max_value - min_value)
    if img.ndim == 3:
        img = img.expand_dims(0)
    if img.shape[-1]==1:
        img = img.repeat(3, axis=-1)
    img[:,10:, :] = 0
    img[:,:, :10]  = 0
    img[:,-10:,:]  = 0
    img[:,:, -10:] = 0
    return img


# Define a dataset that loads your images.
class XrdDataset(Dataset):
    def __init__(self, data_dir, feature_extractor=None, img_blocks=40):
        self.image_dir = data_dir
        self.image_files = get_directories(data_dir)
        self.feature_extractor = feature_extractor
        self.img_blocks = img_blocks
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        print(f"Loading {self.image_files[idx]}")
        img,_ = get_imgs([self.image_files[idx]])
        img = preprocess_images(img)
        img = torch.from_numpy(img).float()
        if self.feature_extractor:
            img = img.squeeze(0)
            img = self.feature_extractor(img, return_tensors="pt").pixel_values
        
        return img

