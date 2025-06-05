### Autoregression Project


## Data
Currently, I amtesting experiment 1027522 with ranges 69-75


### Todos:
- Learn about the preprocessor from hugging face
https://huggingface.co/blog/fine-tune-vit
- Use the preprocessor from hugging face to be able to create an encoder


## Important!!
Ignore file 
data/mfxl1027522_r0030_peaknet.0075.zarr/images


### Requesting a gpu

srun -p gpu-pascal --gres=gpu:1 --pty bash
srun -p gpu-turing --gres=gpu:4 --pty bash
