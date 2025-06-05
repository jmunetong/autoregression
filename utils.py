import time
import uuid
import os

import zarr
import numpy as np
import torch


def load_zarr_files(directory_list, data_id):
    """
    Load Zarr files from a list of directories.

    Args:
        directory_list (list): List of directories containing Zarr files.

    Returns:
        list: List of loaded Zarr arrays.
    """
    zarr_arrays = []
    for directory in directory_list:
        
        if data_id in directory: #TODO: Fix this. This is temporary to be able to run experiments
            # print(directory)
            arr = zarr.open(directory, mode='r')
            zarr_arrays.append(arr)
    return zarr_arrays

def files_to_img(z_arrays, sample_id = None, verbose=False):
    """
    Convert Zarr arrays to images.

    Args:
        z_arrays (list): List of Zarr arrays.

    Returns:
        list: List of images.
    """
    images = []
    num_imges = 0
    for arr in z_arrays:
        img = arr[:] if sample_id is None else arr[sample_id]
        images.append(img)
        num_imges += img.shape[0] if sample_id is None else 1
        if verbose:
            print(f"Loaded image shape: {img.shape}")
    if sample_id is None:
        images = np.concatenate(images, axis=0)
    else:
        images = np.array(images)
    
    assert num_imges == images.shape[0], "Number of images mismatch"
    # if images.ndim == 3:
    #     images = np.expand_dims(images, axis=-1)
    # assert images.ndim == 4, "Images should be 4D after conversion"
    return images



def get_directories(path, sub_dir="images"):
    """"""
    directories  = [os.path.join(path,d, sub_dir) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return directories

def mount_images_to_torch(imgs, device):
    """
    Mounts the images to torch tensors
    """
    imgs = torch.from_numpy(imgs).to(device)
    return imgs

def get_device():
    """
    Get the device to use for torch
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
            
    return device

def create_experiment_id(model_name="vae", data_id=522):
    timestamp = time.strftime("%y%m%d-%H%M")
    unique_id = uuid.uuid4().hex[:8]  # short unique hash
    return f"{timestamp}_{model_name}_d{data_id}_{unique_id}"


def create_directory(model_name, data_id):
    experiment_id = create_experiment_id(model_name, data_id)
    directory = os.path.join("experiments", model_name, experiment_id)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return experiment_id, directory

def print_color(text, color="default"):
    colors = {
        "default": "\033[0m",
        "red":     "\033[91m",
        "green":   "\033[92m",
        "yellow":  "\033[93m",
        "blue":    "\033[94m",
        "magenta": "\033[95m",
        "cyan":    "\033[96m",
        "white":   "\033[97m"
    }

    reset = "\033[0m"
    color_code = colors.get(color.lower(), colors["default"])
    print(f"{color_code}{text}{reset}")