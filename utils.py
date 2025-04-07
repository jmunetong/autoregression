import zarr
import numpy as np
import os

def load_zarr_files(directory_list):
    """
    Load Zarr files from a list of directories.

    Args:
        directory_list (list): List of directories containing Zarr files.

    Returns:
        list: List of loaded Zarr arrays.
    """
    zarr_arrays = []
    for directory in directory_list:
        arr = zarr.open(directory, mode='r')
        zarr_arrays.append(arr)
    return zarr_arrays

def files_to_img(z_arrays):
    """
    Convert Zarr arrays to images.

    Args:
        z_arrays (list): List of Zarr arrays.

    Returns:
        list: List of images.
    """
    images = []
    for arr in z_arrays:
        img = arr[:]
        images.append(img)
    images = np.array(images)
    if images.ndim == 3:
        images = np.expand_dims(images, axis=-1)
    assert images.ndim == 4, "Images should be 4D after conversion"
    return images

def get_imgs(directory_list):
    """
    Get images from Zarr files in the specified directories.

    Args:
        directory_list (list): List of directories containing Zarr files.

    Returns:
        tuple: Tuple containing the list of images and the list of Zarr arrays.
    """
    z_arrays = load_zarr_files(directory_list)
    images = files_to_img(z_arrays)
    return images, z_arrays

def get_directories(path, sub_dir="images"):
    """"""
    directories  = [os.path.join(path, sub_dir) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return directories