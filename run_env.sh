#!/bin/bash
# sbatch /opt/ohpc/pub/examples/python/build_pytorch_anaconda.slurm
srun -p gpu-pascal --gres=gpu:4 --pty bash
conda activate xplr
# srun -p gpu-volta --gres=gpu:2 --pty bash
# srun -p gpu-turing --gres=gpu:2 --pty bash