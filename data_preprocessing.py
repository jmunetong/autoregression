
import os

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import einops

from utils import get_directories, files_to_img, load_zarr_files

def preprocess_images(img:np.ndarray, repeat_dim=False):
    img = (img - img.mean()) / (img.std() + 1e-6)
    if img.ndim == 3:
        img = np.expand_dims(img,-1)
    if img.shape[-1]==1 and repeat_dim:
        img = img.repeat(3, axis=-1)

    img[:,10:, :] = 0
    img[:,:, :10]  = 0
    img[:,-10:,:]  = 0
    img[:,:, -10:] = 0
    return img


# Define a dataset that loads your images.
class XrdDataset(Dataset):
    def __init__(self, data_dir, feature_extractor=None, rescale=False, apply_pooling=True):
        self.zarr_pointers = load_zarr_files(get_directories(data_dir))
        self._preprocess_indeces()
        self.feature_extractor = feature_extractor
        self.rescale = rescale
        self.apply_pool = apply_pooling
        self.avg_pooler = nn.AvgPool2d(kernel_size=(2,2))
    
    def __len__(self):
        return len(self.idx_files)
    
    def _preprocess_indeces(self):  
        self.idx_files = [(i, j) for i, file in enumerate(self.zarr_pointers) for j in range(file.shape[0])]

    def __getitem__(self, idx):

        document_id, sample_id = self.idx_files[idx]
        img = files_to_img([self.zarr_pointers[document_id]], sample_id) 
        img = preprocess_images(img)
        img = torch.from_numpy(img).float()
        
        if self.apply_pool:
            img = einops.rearrange(img, 'b h w c -> b c h w')
            img = self.avg_pooler(img)
            min_dim = min(img.shape[2:])
            img = img[:,:, :min_dim, :min_dim]

        if self.feature_extractor:
            img = img.squeeze(0)
            img = self.feature_extractor(img, return_tensors="pt", do_rescale=False).pixel_values
  
        
        return img.squeeze(0)

