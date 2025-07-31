import os

import json
import yaml
import torch

from torch import  optim

import numpy as np

from transformers import get_cosine_schedule_with_warmup

    
def load_configure_training(state_directory):
    directory = state_directory
    with open(os.path.join(directory, "config.json"), "r") as file:
        model_config = json.load(file)

    with open(os.path.join(directory, "experiment_config.yml"), "r") as file:
        experiment_dict = yaml.safe_load(file)

    return model_config, experiment_dict

def load_diffusers_model(model, directory, accelerator):
    model = model.from_pretrained(directory,use_safetensors=True )
    model = accelerator.prepare(model)
    return model

def load_optim_scheduler(model, directory, accelerator, args):
    """
    Load the optimizer and scheduler state from the specified directory.
    
    Args:
        model: The model to load the optimizer and scheduler for.
        directory (str): The directory where the optimizer and scheduler states are stored.
        accelerator: The accelerator instance for distributed training.
    
    Returns:
        tuple: A tuple containing the loaded optimizer and scheduler.
    """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=1000  # Placeholder, will be updated later
    )
    
    checkpoint = torch.load(os.path.join(directory, "state_dict.pt"), map_location=accelerator.device)

    # TODO: Fix this to be able to load the state_dict.pt file
    if os.path.exists(os.path.join(directory, "optimizer.pt")):
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    if os.path.exists(os.path.join(directory, "scheduler.pt")):
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    optimizer, scheduler = accelerator.prepare(optimizer, scheduler)
    return optimizer, scheduler

def _load_mar_model(checkpoint, model, accelerator):
    model = model.load_state_dict(checkpoint['model'])
    model = accelerator.prepare(model)
    return model


