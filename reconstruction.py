import torch
from utils import get_directories, files_to_img, load_zarr_files
from data_preprocessing import preprocess_images, XrdDataset
from accelerate import init_empty_weights, load_checkpoint_in_model
import matplotlib.pyplot as plt
import numpy as np
import einops
import os
import json
from diffusers import AutoencoderKL
import torch.nn as nn
import umap.umap_ as umap
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.cm as cm
import re
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA



def plot_reconstruction(original: torch.Tensor, reconstructed: torch.Tensor, idx: int, directory: str):
    """
    Plots original and reconstructed images side-by-side.

    Args:
        original (torch.Tensor or np.ndarray): Tensor of shape (B, C, H, W) or (B, H, W)
        reconstructed (torch.Tensor or np.ndarray): Same shape as original
        idx (int): Index in the batch to visualize
    """
    # Convert to numpy if tensors
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu().numpy()
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.detach().cpu().numpy()

    # Handle grayscale or RGB
    if original.ndim == 4:  # (B, C, H, W)
        orig_img = np.transpose(original, (1, 2, 0))
        recon_img = np.transpose(reconstructed, (1, 2, 0))
    elif original.ndim == 3:  # (B, H, W)
        orig_img = original
        recon_img = reconstructed
    else:
        raise ValueError("Expected input shape (B, C, H, W) or (B, H, W)")
    
    # print(f"Original shape: {orig_img.shape}, Reconstructed shape: {recon_img.shape}")  
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(orig_img.squeeze(), cmap='gray' if orig_img.shape[-1] == 1 or orig_img.ndim == 2 else None, vmin = np.percentile(orig_img, 1),
    vmax = np.percentile(orig_img, 99))
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(recon_img.squeeze(), cmap='gray' if recon_img.shape[-1] == 1 or recon_img.ndim == 2 else None, vmin = np.percentile(recon_img, 1),
    vmax = np.percentile(recon_img, 99))
    axes[1].set_title("Reconstructed")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(directory, f'dir{idx}_.png'))

# ========== Config ==========
model_name = "vae_kl_local_contrast" # "vq" or "vae_kl" or "vae_kl_direct_contrast" or "vae_kl_local_contrast"
model_path = "experiments/vae_kl/250529-2116_vae_kl_local_contrast_d522_7ab136f8"
weights_path = f"{model_path}/diffusion_pytorch_model.safetensors"
data_dir = "data"
data_id = "422"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_dir = "."
batch_size = 8

# ========== Identify .zarr groups ==========
zarr_dirs = get_directories(data_dir)
print("zarr_dirs:", zarr_dirs)
zarr_files = load_zarr_files(zarr_dirs, data_id=data_id)

group_labels = []
group_names = []
for i, (zf, zdir) in enumerate(zip(zarr_files, zarr_dirs)):
    count = zf.shape[0]  # Suppose we have 40 images in each group
    group_labels.extend([i] * count)

    # data/mfxl1025422_r0309_peaknet.0000.zarr/images -> group_name = 0000
    match = re.search(r'peaknet\.(\d{4})\.zarr', zdir)
    if match:
        group_names.append(match.group(1))
    else:
        raise ValueError(f"Could not extract group name from path: {zdir}")



# ========== Load Dataset ==========
dataset = XrdDataset(data_dir=data_dir, data_id=data_id, apply_pooling=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# ========== Load Model ==========
config_path = os.path.join(model_path, "config.json")
with open(config_path, "r") as f:
    config_dict = json.load(f)
vae = AutoencoderKL(**config_dict)
vae.to(device)
vae.eval()

# ========== Latent Extraction ==========
latents = []
first_original = None
first_recon = None

for i, batch in enumerate(tqdm(dataloader, desc="Extracting latents")):
    batch = batch.to(device)

    with torch.no_grad():
        latent = vae.encode(batch).latent_dist.mean   # shape: (B, latent_dim, h, w)
        flat_latent = latent.view(latent.size(0), -1)  # flatten spatial dims
        latents.append(flat_latent.cpu().numpy())

        # Save first original and reconstructed images
        if i == 0:
            recon = vae.decode(latent).sample
            first_original = batch[0].cpu()
            first_recon = recon[0].cpu()

latents = np.concatenate(latents, axis=0)  # shape: (N, latent_dim_flat)

# ========== Save First Reconstruction ==========
if first_original is not None and first_recon is not None:
    plot_reconstruction(first_original, first_recon, idx=0, directory=save_dir)

# ========== UMAP ==========
# Construct hover text for each point
hover_texts = []
for group_idx, zf in enumerate(zarr_files):
    for i in range(zf.shape[0]):
        hover_texts.append(f"group_{group_names[group_idx]}/image_{i:04d}")

colors = cm.get_cmap('tab10', len(zarr_files))

group_labels = np.array(group_labels)  # shape: (N,)

# ========== UMAP 2D Interactive ==========
reducer_2d = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42, n_components=2)
embedding_2d = reducer_2d.fit_transform(latents)

fig_2d = px.scatter(
    x=embedding_2d[:, 0],
    y=embedding_2d[:, 1],
    color=[f"group_{group_names[i]}" for i in group_labels],
    hover_name=hover_texts,
    title="Interactive 2D UMAP of Latent Space"
)
fig_2d.update_traces(marker=dict(size=5, opacity=0.8))
fig_2d.write_html(os.path.join(save_dir, f"{model_name}_umap_latents_2d_interactive.html"))

# ========== UMAP 3D Interactive ==========
reducer_3d = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42, n_components=3)
embedding_3d = reducer_3d.fit_transform(latents)
fig_3d = go.Figure()

for i, group_name in enumerate(group_names):
    indices = np.where(group_labels == i)[0]
    fig_3d.add_trace(go.Scatter3d(
        x=embedding_3d[indices, 0],
        y=embedding_3d[indices, 1],
        z=embedding_3d[indices, 2],
        mode='markers',
        marker=dict(size=3, opacity=0.8),
        name=f"group_{group_name}",
        text=[hover_texts[idx] for idx in indices],
        hoverinfo='text'
    ))

fig_3d.update_layout(
    title="Interactive 3D UMAP of Latent Space",
    scene=dict(
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        zaxis_title="UMAP 3"
    ),
    legend=dict(font=dict(size=10))
)
fig_3d.write_html(os.path.join(save_dir, f"{model_name}_umap_latents_3d_interactive.html"))

# ---------- PCA ----------
pca_2d = PCA(n_components=2)
embedding_pca_2d = pca_2d.fit_transform(latents)

pca_3d = PCA(n_components=3)
embedding_pca_3d = pca_3d.fit_transform(latents)

# ========== 2D PCA Plot ==========
fig_pca_2d = px.scatter(
    x=embedding_pca_2d[:, 0],
    y=embedding_pca_2d[:, 1],
    color=[f"group_{group_names[i]}" for i in group_labels],
    hover_name=hover_texts,
    title="PCA (2D) of Latent Space"
)
fig_pca_2d.update_traces(marker=dict(size=5, opacity=0.8))
fig_pca_2d.write_html(os.path.join(save_dir, f"{model_name}_pca_latents_2d_interactive.html"))

# ========== 3D PCA Plot ==========
fig_pca_3d = go.Figure()

for i, group_name in enumerate(group_names):
    indices = np.where(group_labels == i)[0]
    fig_pca_3d.add_trace(go.Scatter3d(
        x=embedding_pca_3d[indices, 0],
        y=embedding_pca_3d[indices, 1],
        z=embedding_pca_3d[indices, 2],
        mode='markers',
        marker=dict(size=3, opacity=0.8),
        name=f"group_{group_name}",
        text=[hover_texts[idx] for idx in indices],
        hoverinfo='text'
    ))

fig_pca_3d.update_layout(
    title="PCA (3D) of Latent Space",
    scene=dict(
        xaxis_title="PCA 1",
        yaxis_title="PCA 2",
        zaxis_title="PCA 3"
    ),
    legend=dict(font=dict(size=10))
)
fig_pca_3d.write_html(os.path.join(save_dir, f"{model_name}_pca_latents_3d_interactive.html"))


#################################
#2D Plot

# plt.figure(figsize=(8, 6))
# for i in range(len(zarr_files)):
#     indices = group_labels == i
#     plt.scatter(embedding_2d[indices, 0], embedding_2d[indices, 1], s=10, label=group_names[i], alpha=0.7, color=colors(i))
# plt.title("2D UMAP of Latent Space")
# plt.xlabel("UMAP 1")
# plt.ylabel("UMAP 2")
# plt.legend(fontsize=8, loc="best")
# plt.grid(True)
# plt.savefig(os.path.join(save_dir, "umap_latents_2d_baseline.png"))

# # ========== UMAP 3D Plot ==========
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')
# for i in range(len(zarr_files)):
#     indices = group_labels == i
#     ax.scatter(embedding_3d[indices, 0], embedding_3d[indices, 1], embedding_3d[indices, 2],
#                s=10, label=group_names[i], alpha=0.7, color=colors(i))
# ax.set_title("3D UMAP of Latent Space")
# ax.set_xlabel("UMAP 1")
# ax.set_ylabel("UMAP 2")
# ax.set_zlabel("UMAP 3")
# ax.legend(fontsize=8, loc='best')
# plt.tight_layout()
# plt.savefig(os.path.join(save_dir, "umap_latents_3d_baseline.png"))

# # ========== Encode & Decode ==========
# img_tensor = img_tensor.to(device)
# with torch.no_grad():
#     latent = vae.encode(img_tensor).latent_dist.sample()
#     print(f"Latent shape: {latent.shape}")
#     recon = vae.decode(latent).sample
#     print(f"Reconstructed shape: {recon.shape}")

# # ========== Anti-Standardization ==========
# img_tensor_denorm = img_tensor * std + mean
# recon_denorm = recon * std + mean

# # ========== Visualize ==========
# plot_reconstruction(img_tensor_denorm.squeeze(0), recon_denorm.squeeze(0), idx=0, directory=save_dir)