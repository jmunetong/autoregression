
import os

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np

from utils import get_directories, files_to_img, load_zarr_files

def preprocess_images(img:np.ndarray):
    img = (img - img.mean()) / (img.std() + 1e-6)
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
    def __init__(self, data_dir, feature_extractor=None, rescale=False, img_blocks=40):
        self.image_dir = data_dir
        self.image_files = get_directories(data_dir)
        print(type(self.image_files))
        print(self.image_files)
        self.zarr_pointers = load_zarr_files(self.image_files)
        self._preprocess_indeces()
        self.feature_extractor = feature_extractor
        self.img_blocks = img_blocks
        self.rescale = rescale
    
    def __len__(self):
        return len(self.idx_files)
    
    def _preprocess_indeces(self):  
        self.idx_files = [(i, j) for i, file in enumerate(self.zarr_pointers) for j in range(file.shape[0])]

    def __getitem__(self, idx):
        print(f'This is idx number:{idx}')
        print(f"Loading {self.image_files[idx]}")
        document_id, sample_id = self.idx_files[idx]
        img = files_to_img([self.zarr_pointers[document_id]], sample_id) # TODO: 
        img = preprocess_images(img)
        img = torch.from_numpy(img).float()
        print(img.std())
        print(img.mean())
        if self.feature_extractor:
            img = img.squeeze(0)
            print(f'this is the type of image: {type(img)}')
            img = self.feature_extractor(img, return_tensors="pt", do_rescale=False).pixel_values
        print(img.std())
        print(img.mean()) 
        return img

