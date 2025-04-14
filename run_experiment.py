import os

import zarr
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import einops
from transformers import ViTForImageClassification, ViTImageProcessor, ViTFeatureExtractor, ViTConfig, ViTModel, pipeline
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader

from utils import get_directories, get_imgs, get_device
from data_preprocessing import preprocess_images,XrdDataset


FEATURE_EXTRACTOR_PATH = "google/vit-base-patch16-224"
DATA_PATH = "data"
URL_MODEL = "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors"

def run(args):
    feature_extractor = ViTImageProcessor.from_pretrained(FEATURE_EXTRACTOR_PATH)
    dataset = XrdDataset(data_dir=args.data_path,    feature_extractor=feature_extractor)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    device = get_device()
    print(f'Current CUDA device is:{device}')
    vae = AutoencoderKL.from_single_file(URL_MODEL)
    vae.to(device)
    vae.train()

    optimizer = optim.Adam(vae.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        for i, batch in enumerate(dataloader):
            if batch.ndim > 4:
                batch = einops.rearrange(batch, 'd b c h w -> (d b) c h w')

            batch = batch.to(device)
            optimizer.zero_grad()
            recon_batch = vae(batch)
            loss = criterion(recon_batch, batch)
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print(f"Batch {i}, Loss: {loss.item()}")

    torch.save(vae.state_dict(), "vae_model.pth")

        


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default=DATA_PATH, help="Path to the data directory")
    parser.add_argument("--batch_size", "-b", type=int, default=1, help="Batch size for training")
    parser.add_argument("--num_epochs", "-e", type=int, default=10, help="Number of epochs for training")
    args = parser.parse_args()
    run(args)