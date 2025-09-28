import time
import uuid
import os

from omegaconf import DictConfig
import zarr
import numpy as np
import torch
import yaml
import json 

from plot import plot_reconstruction, plot_diff, transform_to_image


def is_wrapped(model, accelerator):
    return accelerator.unwrap_model(model) if hasattr(model, 'module') else model

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
    # Convert args to dictionary and create a copy
    metadata = vars(args).copy()
    # Add any computed or additional fields
    metadata["input_shape"] = None
    metadata["latent_shape"] = None
    
    return metadata


def create_experiment_id(model_name="vae", data_id=522):
    timestamp = time.strftime("%y%m%d-%H%M")
    unique_id = uuid.uuid4().hex[:8]  # short unique hash
    return f"{timestamp}_{model_name}_d{data_id}_{unique_id}"


def create_directory(output_path, model_name, data_id):
    #TODO: FIX THIS SINCE WE ARE TRYING TO ACCESS THE MODEL FROM THIS SETUP 
    
    # model_name = f'{model_name}_d{data_id}'
    # directory = os.path.join(output_path, model_name)
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    # return directory
    os.makedirs(output_path, exist_ok=True)
    return output_path
    

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


def prepare_state_dict(args, accelerator, output_path):
    # Make sure that args.diff and args.prettrained_vae are usually used so that VAE does not need to be trained from scratch. 
    if args.latent_diff and args.pretrained_vae_path is not None:
        with open(os.path.join(args.pretrained_vae_path, "experiment_config.yml"), "r") as file:
            state_dict = yaml.safe_load(file)
        update_args(args,state_dict )

    # information for saving model-experiment characteristics.
    md_name = args.model_name if not args.diff else diff_name_config(args.latent_diff, args)
    model_name_dir = md_name if not args.test_pipeline else f"{md_name}_test"
    torch.cuda.empty_cache()
    # Configure training

    # (args, output_path, model_name_dir, accelerator)
    directory, experiment_dict = configure_training(args,output_path,model_name_dir, accelerator)
    return directory, experiment_dict

def diff_name_config(use_vae, args):
    return f"diff_{args.model_name}" if use_vae else f"diff_non_vae_{args.model_name}"


def is_experiment_from_scratch(args):
    return args.train_vae_from_scratch or args.train_diff_from_scratch
    

def is_scratch_training(args):
    return args.vae_from_scratch or args.diff_from_scratch

def determine_experiment_directory(args):
    if args.pretrained_diff_path is not None and args.train_diff_from_checkpoint:
        directory = args.pretrained_diff_path
    else:
        directory = args.pretrained_vae_path

    assert directory is not None

    return directory

def configure_training(args, output_path, model_name_dir, accelerator):
    # Step 1: Main process creates directory and metadata
    if accelerator.is_main_process:
        if is_experiment_from_scratch(args):
            directory = create_directory(output_path, model_name_dir, args.data_id)
            experiment_dict = build_experiment_metadata(args)
        else:
            metadata_path = determine_experiment_directory(args)
            with open(os.path.join(metadata_path, "metadata.json"), 'r') as file:
                d = json.load(file)
            directory = d["directory"]
            experiment_dict = d['experiment_dict']
            del d, metadata_path

    
        # Convert directory to string if it's a Path object
        directory_str = str(directory) if hasattr(directory, "__fspath__") else directory
        
        # Store in temporary file 
        temp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_dir_path.txt")
        with open(temp_path, "w") as f:
            f.write(directory_str)
        
        # Store the metadata in the actual directory

        with open(f"{directory}/metadata.json", "w") as f:
            json.dump({
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

    return directory, experiment_dict
    
def create_args_compatibility(cfg: DictConfig):
    """Convert Hydra config to args-like object for backward compatibility"""
    class Args:
        def __getattr__(self, name):
            # Fallback for any missing attributes
            print(f"Warning: Accessing undefined attribute '{name}', returning None")
            return None
            
        def __setattr__(self, name, value):
            # Allow setting new attributes
            super().__setattr__(name, value)
    
    args = Args()
    
    # Model parameters
    args.model_name = cfg.model.model_name
    args.latent_channels = getattr(cfg.model, 'latent_channels', 4)
    
    # Training parameters
    args.batch_size = cfg.data.batch_size
    args.test_pipeline = cfg.experiment_type.test_pipeline
    args.num_epochs = cfg.experiment_type.num_epochs
    args.lr = cfg.experiment_type.lr
    args.weight_decay = cfg.experiment_type.weight_decay
    args.beta_recons = cfg.experiment_type.beta_recons
    args.recons_loss = cfg.experiment_type.recons_loss
    args.alpha_mse = cfg.experiment_type.alpha_mse
    args.ema_decay = cfg.experiment_type.ema_decay
    
    # Training mode flags
    args.train_vae_from_checkpoint = cfg.experiment_type.train_vae_from_checkpoint
    args.train_vae_from_scratch = cfg.experiment_type.train_vae_from_scratch
    args.train_diff_from_checkpoint = cfg.experiment_type.train_diff_from_checkpoint
    args.train_diff_from_scratch = cfg.experiment_type.train_diff_from_scratch
    args.pretrained_vae_path = cfg.experiment_type.pretrained_vae_path
    args.pretrained_diff_path = cfg.experiment_type.pretrained_diff_path
    args.use_annealing = cfg.experiment_type.use_annealing
    args.annealing_shape = cfg.experiment_type.annealing_shape
    
    # Data parameters
    args.data_id = cfg.data.data_id
    args.avg_pooling = cfg.data.avg_pooling
    args.topk = cfg.data.topk
    args.data_path = cfg.data.data_path
    args.train_ratio = cfg.data.train_ratio
    args.seed = cfg.data.seed
    
    # Inference parameters
    args.inference = cfg.inference.inference
    args.generate_samples = cfg.inference.generate_samples
    args.num_samples = cfg.inference.num_samples
    
    # Diffusion parameters - handle both explicit and inferred flags
    args.diff = cfg.model.model_name == "diff"
    args.latent_diff = getattr(cfg.model, 'latent_diff', cfg.model.model_name == "latent_diff")
    
    # Diffusion model parameters
    args.diff_epochs = getattr(cfg.model, 'diff_epochs', 10)
    args.patch_size = getattr(cfg.model, 'patch_size', 16)
    args.vit_size = getattr(cfg.model, 'vit_size', 'base')
    if args.diff:
        args.diff_args = cfg.model.diffusion_kwargs 
        assert isinstance(args.diff_args, dict), "Incorrect type must be a dictionary for arguments to be passed into model"

    # Legacy compatibility attributes (in case they're referenced elsewhere)
    args.vae_from_scratch = args.train_vae_from_scratch
    args.diff_from_scratch = args.train_diff_from_scratch
    
    return args

