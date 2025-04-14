#!/bin/bash
sbatch /opt/ohpc/pub/examples/python/build_pytorch_anaconda.slurm
conda env create -f env.yml