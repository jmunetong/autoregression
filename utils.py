import time
import uuid
import os

import zarr
import numpy as np
import torch
import yaml
import json 

from plot import plot_reconstruction, plot_diff, transform_to_image


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


def build_experiment_metadata(args):
    metadata = {
        "model_name": args.model_name,
        "data_id": args.data_id,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "beta_recons": args.beta_recons,
        "recons_loss": args.recons_loss,
        "input_shape": None,
        "latent_shape": None,
        "avg_pooling": args.avg_pooling,
        "weight_decay": args.weight_decay,
        "learning_rate": args.lr,
        "latent_channels": args.latent_channels,

    }
    if args.recons_loss == 'iwmse':
        metadata['alpha_mse'] = args.alpha_mse  
    return metadata

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

def update_args(args, state_dict):
    for key, value in state_dict.items():
        if hasattr(args, key):
            setattr(args, key, value)


def prepare_state_dict(args, accelerator):
    # Make sure that args.diff and args.prettrained_vae are usually used so that VAE does not need to be trained from scratch. 
    if args.diff and args.pretrained_vae:
        with open(os.path.join(args.pretrained_vae, "experiment_config.yml"), "r") as file:
            state_dict = yaml.safe_load(file)
        update_args(args,state_dict )

    # information for saving model-experiment characteristics.
    md_name = args.model_name if not args.diff else diff_name_config(args.use_vae, args)
    model_name_dir = md_name if not args.test_pipeline else f"{md_name}_test"
    torch.cuda.empty_cache()

    # Configure training
    model_id, directory, experiment_dict = configure_training(args, model_name_dir, accelerator)
    return model_id, directory, experiment_dict

def diff_name_config(use_vae, args):
    return f"diff_{args.model_name}" if use_vae else f"diff_non_vae_{args.model_name}"

def generate_vae_samples(model, dataloader, directory):
    count = 0
    for i, batch in enumerate(dataloader):
        if i > 2:
            break
        for j in range(batch.shape[0]):
            recons = model(batch[j].unsqueeze(0), return_dict=True).sample
            plot_reconstruction(batch[j],recons, idx=count, directory=directory)
            count += 1
            del recons
            if count > 5:
                break

def generate_diff_samples(model, diff_model, directory, count=1, encoding_shape=None, image_shape=None, min_pixel=0, max_pixel=1, use_vae=False):
    for i in range(count):
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

def configure_training(args, model_name_dir, accelerator):
        # Step 1: Main process creates directory and metadata
    if accelerator.is_main_process:
        model_id, directory = create_directory(model_name_dir, args.data_id)
        experiment_dict = build_experiment_metadata(args)
        
        # Convert directory to string if it's a Path object
        directory_str = str(directory) if hasattr(directory, "__fspath__") else directory
        
        # Store in temporary file 
        temp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_dir_path.txt")
        with open(temp_path, "w") as f:
            f.write(directory_str)
        
        # Store the metadata in the actual directory
        with open(f"{directory}/metadata.json", "w") as f:
            json.dump({
                "model_id": model_id,
                "directory": directory_str,
                "experiment_dict": experiment_dict
            }, f)

    # Step 2: Wait for file to be written
    accelerator.wait_for_everyone()

    # Step 3: All non-main processes read the directory path and load values
    if not accelerator.is_main_process:
        temp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_dir_path.txt")
        with open(temp_path, "r") as f:
            directory = f.read().strip()
        
        with open(f"{directory}/metadata.json", "r") as f:
            data = json.load(f)
            model_id = data["model_id"]
            experiment_dict = data["experiment_dict"]

    # Clean up the temporary file
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        temp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_dir_path.txt")
        device = get_device()
        print(f'Current CUDA device is:{device}')
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if args.test_pipeline:
            print_color("Test pipeline is enabled. Exiting.", "red")
        else:
            print_color("Experiment Running", "Green")

    return model_id, directory, experiment_dict
    