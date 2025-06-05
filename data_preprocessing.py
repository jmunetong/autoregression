
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

    img[:,:10, :] = 0
    img[:,:, :10]  = 0
    img[:,-10:,:]  = 0
    img[:,:, -10:] = 0
    return img


# Define a dataset that loads your images.
class XrdDataset(Dataset):
    def __init__(self, data_dir, data_id, feature_extractor=None, rescale=False, apply_pooling=False, top_k=1.0, filter_size=3):
        self.data_id = data_id
        self.zarr_pointers = load_zarr_files(get_directories(data_dir), data_id=data_id)
        self.top_k = int(len(self.zarr_pointers) * min(top_k,1.0))
        self.zarr_pointers = self.zarr_pointers[:self.top_k]
        self._preprocess_indeces()
        self.feature_extractor = feature_extractor
        self.rescale = rescale
        self.apply_pool = apply_pooling
        self.filter_size = filter_size #TODO: ADD PARAMETER TODO THIS
        self.avg_pooler = nn.AvgPool2d(kernel_size=(self.filter_size,self.filter_size))
        self.i = 0
        self.min = None
        self.max = None
        
    def __len__(self):
        return len(self.idx_files)
    
    def get_image_shape(self):
        sample = self.__getitem__(0)  # Get the shape of the first image
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
        # current image of shape [1,1, 1667, 1665], and I want it to be [1,1,1664, 1664]
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

        return img.squeeze(0)



