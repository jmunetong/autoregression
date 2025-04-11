import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import zarr
import os
import einops

from transformers import ViTForImageClassification, ViTImageProcessor, ViTFeatureExtractor, ViTConfig, ViTModel, pipeline
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader

from utils import get_directories, get_imgs, get_device
from data_preprocessing import preprocess_images,XrdDataset


FEATURE_EXTRACTOR_PATH = "google/vit-base-patch16-224"
DATA_PATH = "data"
URL_MODEL = "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors"

def run(data_path):
    feature_extractor = ViTImageProcessor.from_pretrained(FEATURE_EXTRACTOR_PATH)
    dataset = XrdDataset(data_dir=data_path,    feature_extractor=feature_extractor)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    device = get_device()
    vae = AutoencoderKL.from_single_file(URL_MODEL)
    vae.to(device)
    vae.train()
    optimizer = optim.Adam(vae.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    

if __name__ == '__main__':
    run(DATA_PATH)