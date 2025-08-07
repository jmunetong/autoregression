import torch
from torch.utils.data import Dataset, random_split, Subset
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import einops
from sklearn.model_selection import train_test_split
import random

from utils import get_directories, files_to_img, load_zarr_files

def preprocess_images(img:np.ndarray, repeat_dim=False):
    img = (img - img.mean()) / (img.std() + 1e-6)
    if img.ndim == 3:
        img = np.expand_dims(img,-1)
    if img.shape[-1]==1 and repeat_dim:
        img = img.repeat(3, axis=-1)

    img[:,:10, :] = 0
    img[:,:, :10]  = 0
    img[:,-10:,:]  = 0
    img[:,:, -10:] = 0
    return img

def pad_to_multiple(x, multiple=16):
    h, w = x.shape[-2], x.shape[-1]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    return F.pad(x, (0, pad_w, 0, pad_h), "constant", 0)

class XrdDataset(Dataset):
    def __init__(self, data_dir, data_id, feature_extractor=None,       rescale=False, 
                 apply_pooling=False, top_k=1.0, filter_size=3, zarr_indices=None):
        self.data_id = data_id
        self.zarr_pointers = load_zarr_files(get_directories(data_dir), data_id=data_id)
        
        # If specific zarr indices are provided (for train/val split), use only those
        if zarr_indices is not None:
            self.zarr_pointers = [self.zarr_pointers[i] for i in zarr_indices]
        
        self.top_k = int(len(self.zarr_pointers) * min(top_k,1.0))
        self.zarr_pointers = self.zarr_pointers[:self.top_k]
        self._preprocess_indeces()
        self.feature_extractor = feature_extractor
        self.rescale = rescale
        self.apply_pool = apply_pooling
        self.filter_size = filter_size
        self.avg_pooler = nn.AvgPool2d(kernel_size=(self.filter_size,self.filter_size))
        self.i = 0
        self.min = None
        self.max = None
        
    def __len__(self):
        return len(self.idx_files)
    
    def get_image_shape(self):
        sample = self.__getitem__(0)
        return sample.shape
    
    def _preprocess_indeces(self):  
        self.idx_files = [(i, j) for i, file in enumerate(self.zarr_pointers) for j in range(file.shape[0])]
    
    def get_min_max(self):
        return self.min, self.max
    
    def __getitem__(self, idx):
        document_id, sample_id = self.idx_files[idx]
        img = files_to_img([self.zarr_pointers[document_id]], sample_id) 
        img = preprocess_images(img)
        img = torch.from_numpy(img).float()
        img = einops.rearrange(img, 'b h w c -> b c h w')
        
        if self.min is None or self.max is None:
            self.min = np.percentile(img, 1)
            self.max = np.percentile(img, 99)
            
        if self.apply_pool:
            img = self.avg_pooler(img)
            if img.shape[-1] != img.shape[-2]:
                min_dim = min(img.shape[2:])
                img = img[:,:, :min_dim, :min_dim]
            if self.filter_size == 3 and img.shape[-1] == 555:
                img = img[:,:, 1:-2, 1:-2]
        else:
            img = img[:,:,3:, :-1] if img.shape[-1] != img.shape[-2] else img
        
        img = pad_to_multiple(img, 16)
        return img.squeeze(0)

# Method 1: Split at zarr file level (recommended for your use case)
def create_train_val_datasets_zarr_split(data_dir, data_id, train_ratio=0.8, 
                                        random_seed=42, **dataset_kwargs):
    """
    Split at the zarr file level to ensure samples from the same file 
    stay in the same split (train or validation).
    """
    # Get all zarr files
    all_zarr_pointers = load_zarr_files(get_directories(data_dir), data_id=data_id)
    zarr_indices = list(range(len(all_zarr_pointers)))
    
    # Split zarr file indices
    random.seed(random_seed)
    random.shuffle(zarr_indices)
    
    split_point = int(len(zarr_indices) * train_ratio)
    train_zarr_indices = zarr_indices[:split_point]
    val_zarr_indices = zarr_indices[split_point:]
    #TODO: we need to add accelerator here so that it only opens the unique zar files idx and then loads images
    # TODO: print this only if in main accelerator
    print(f"Train zarr files: {len(train_zarr_indices)}, Validation zarr files: {len(val_zarr_indices)}")
    
    # Create datasets with specific zarr files
    train_dataset = XrdDataset(data_dir, data_id, zarr_indices=train_zarr_indices, **dataset_kwargs)
    val_dataset = XrdDataset(data_dir, data_id, zarr_indices=val_zarr_indices, **dataset_kwargs)
    
    return train_dataset, val_dataset

