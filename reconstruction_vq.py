import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import json
import re
from tqdm import tqdm
from sklearn.decomposition import PCA
import umap.umap_ as umap
import plotly.express as px
import plotly.graph_objects as go
from safetensors.torch import load_file

from diffusers import VQModel
from utils import get_directories, load_zarr_files
from data_preprocessing import XrdDataset


# ----------- Config -----------
model_path = "experiments/vq/250513-1921_vq_d422_75cb5b0b"
weights_path = f"{model_path}/diffusion_pytorch_model.safetensors"
data_dir = "data"
data_id = "422"
save_dir = "."
batch_size = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------- Get zarr groups & group labels -----------
zarr_dirs = get_directories(data_dir)
zarr_files = load_zarr_files(zarr_dirs, data_id=data_id)

group_labels = []
group_names = []
for i, (zf, zdir) in enumerate(zip(zarr_files, zarr_dirs)):
    count = zf.shape[0]
    group_labels.extend([i] * count)
    match = re.search(r'peaknet\.(\d{4})\.zarr', zdir)
    if match:
        group_names.append(match.group(1))
    else:
        raise ValueError(f"Could not extract group name from path: {zdir}")

group_labels = np.array(group_labels)


# ----------- Load dataset -----------
dataset = XrdDataset(data_dir=data_dir, data_id=data_id, apply_pooling=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


# ----------- Load VQModel -----------
config_path = os.path.join(model_path, "config.json")
with open(config_path, "r") as f:
    config_dict = json.load(f)

model = VQModel(**config_dict)
state_dict = load_file(weights_path)
model.load_state_dict(state_dict)
model.to(device)
model.eval()


# ----------- Latent extraction -----------
latents = []
hover_texts = []

idx = 0
with torch.no_grad():
    for batch in tqdm(dataloader, desc="Extracting VQ latents"):
        batch = batch.to(device)
        latent = model.encode(batch).latents  # shape: (B, C, H, W)
        flat_latent = latent.view(latent.size(0), -1)
        latents.append(flat_latent.cpu().numpy())

        for i in range(batch.shape[0]):
            group_idx = group_labels[idx]
            hover_texts.append(f"group_{group_names[group_idx]}/image_{idx % 40:04d}")
            idx += 1

latents = np.concatenate(latents, axis=0)


# ----------- PCA 2D -----------
pca_2d = PCA(n_components=2).fit_transform(latents)
fig_pca_2d = px.scatter(
    x=pca_2d[:, 0],
    y=pca_2d[:, 1],
    color=[f"group_{group_names[i]}" for i in group_labels],
    hover_name=hover_texts,
    title="VQModel PCA (2D)"
)
fig_pca_2d.update_traces(marker=dict(size=5, opacity=0.8))
fig_pca_2d.write_html(os.path.join(save_dir, "vq_pca_2d.html"))


# ----------- PCA 3D -----------
pca_3d = PCA(n_components=3).fit_transform(latents)
fig_pca_3d = go.Figure()
for i, name in enumerate(group_names):
    indices = np.where(group_labels == i)[0]
    fig_pca_3d.add_trace(go.Scatter3d(
        x=pca_3d[indices, 0], y=pca_3d[indices, 1], z=pca_3d[indices, 2],
        mode="markers", marker=dict(size=3), name=f"group_{name}",
        text=[hover_texts[j] for j in indices], hoverinfo="text"
    ))
fig_pca_3d.update_layout(
    title="VQModel PCA (3D)",
    scene=dict(xaxis_title="PCA 1", yaxis_title="PCA 2", zaxis_title="PCA 3")
)
fig_pca_3d.write_html(os.path.join(save_dir, "vq_pca_3d.html"))


# ----------- UMAP 2D -----------
umap_2d = umap.UMAP(n_components=2).fit_transform(latents)
fig_umap_2d = px.scatter(
    x=umap_2d[:, 0],
    y=umap_2d[:, 1],
    color=[f"group_{group_names[i]}" for i in group_labels],
    hover_name=hover_texts,
    title="VQModel UMAP (2D)"
)
fig_umap_2d.update_traces(marker=dict(size=5, opacity=0.8))
fig_umap_2d.write_html(os.path.join(save_dir, "vq_umap_2d.html"))


# ----------- UMAP 3D -----------
umap_3d = umap.UMAP(n_components=3).fit_transform(latents)
fig_umap_3d = go.Figure()
for i, name in enumerate(group_names):
    indices = np.where(group_labels == i)[0]
    fig_umap_3d.add_trace(go.Scatter3d(
        x=umap_3d[indices, 0], y=umap_3d[indices, 1], z=umap_3d[indices, 2],
        mode="markers", marker=dict(size=3), name=f"group_{name}",
        text=[hover_texts[j] for j in indices], hoverinfo="text"
    ))
fig_umap_3d.update_layout(
    title="VQModel UMAP (3D)",
    scene=dict(xaxis_title="UMAP 1", yaxis_title="UMAP 2", zaxis_title="UMAP 3")
)
fig_umap_3d.write_html(os.path.join(save_dir, "vq_umap_3d.html"))
