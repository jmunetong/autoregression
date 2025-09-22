#!/bin/bash
#SBATCH --account=mph121
#SBATCH --partition=batch
#SBATCH --qos=debug
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=8
#SBATCH --job-name=test_multi_gpu
#SBATCH --output=slurm/multi_gpu-%j.out
#SBATCH --error=slurm/multi_gpu-%j.err
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=jmuneton@stanford.edu


EXPERIMENT_TYPE=${1:-train}

# Validate the argument
if [[ "$EXPERIMENT_TYPE" != "test" && "$EXPERIMENT_TYPE" != "train" ]]; then
    echo "Error: experiment_type must be 'test' or 'train'"
    echo "Usage: $0 [test|train]"
    exit 1
fi

echo "Running experiment with type: $EXPERIMENT_TYPE"
mkdir -p slurm

echo "Starting multi-GPU job on: $(hostname)"

# Load modules
# module purge
# Load required modules for Frontier
module load PrgEnv-amd
module load rocm/6.2.4
module load cray-mpich
# module load miniforge3/23.11.0-0
# Set up environment for AMD GPUs - let SLURM handle GPU assignment
# export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # All 8 GCDs per node
export MPICH_GPU_SUPPORT_ENABLED=1            # Enable GPU-aware MPI
export FI_MR_CACHE_MONITOR=memhooks            # Recommended for GPU-aware MPI
export FI_CXI_RX_MATCH_MODE=software           # Recommended for GPU-aware MPI

# PyTorch/ROCm environment variables
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:512
export HIP_FORCE_DEV_KERNARG=1
export NCCL_DEBUG=INFO
export TORCH_NCCL_BLOCKING_WAIT=0

# Fix "Address family not supported by protocol" errno 97 errors (critical for Frontier)
export NCCL_SOCKET_FAMILY=AF_INET  # Force IPv4-only communication
export NCCL_IB_DISABLE=1           # Disable InfiniBand if causing issues


# Configure MIOpen cache to avoid SQLite database disk I/O errors (ORNL Frontier fix)
export MIOPEN_USER_DB_PATH="/tmp/my-miopen-cache-$USER-$SLURM_JOB_ID"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

# # Use explicit paths
# CONDA_ENV_PATH="/ccs/home/jmuneton/.conda/envs/pytorch_env"
# # export PATH="$CONDA_ENV_PATH/bin:$PATH"
# conda activate $CONDA_ENV_PATH


# Set explicit paths
# CONDA_ENV_PATH="/ccs/home/jmuneton/.conda/envs/pytorch_env"
# export PATH="$CONDA_ENV_PATH/bin:$PATH"
# export PYTHONPATH="$CONDA_ENV_PATH/lib/python3.10/site-packages:$PYTHONPATH"

# WandB configuration
export WANDB_MODE=offline
export WANDB_DIR="$HOME/wandb_offline_logs"
export WANDB_API_KEY="bbd9552411458167539449b8da39ae4718b9617a"
mkdir -p $HOME/wandb_offline_logs

# # GPU environment for multi-GPU
# export ROCM_PATH=${ROCM_PATH:-/opt/rocm}
# export LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/lib64:$LD_LIBRARY_PATH
# export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:512
# export HSA_OVERRIDE_GFX_VERSION=9.0.0
# export OMP_NUM_THREADS=7

# # CRITICAL: Force Gloo backend for AMD GPUs
# export TORCH_DISTRIBUTED_BACKEND=gloo
# export NCCL_DISABLED=1
# export TORCH_NCCL_BLOCKING_WAIT=0

# Network setup for distributed training
NODE_NAME=$(hostname -s)
export MASTER_ADDR=$NODE_NAME
export MASTER_PORT=29500

echo "=== Multi-GPU Configuration ==="
echo "Hostname: $NODE_NAME"
echo "SLURM_GPUS_PER_NODE: $SLURM_GPUS_PER_NODE"
echo "Backend: $TORCH_DISTRIBUTED_BACKEND"
echo "Visible devices: $CUDA_VISIBLE_DEVICES"

# Verify environment
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
"

# Run with accelerate launch for multi-GPU
ACCELERATE_CONFIG_FILE="accelerate_config/multigpu_config.yaml"
accelerate launch --config_file $ACCELERATE_CONFIG_FILE \
    run_hydra_experiment.py \
    model=vae_kl \
    experiment_type=$EXPERIMENT_TYPE \
    experiment_type.recons_loss=iwmse \
    data.data_id=522

echo "Multi-GPU experiment completed"
echo "Logs saved to: $WANDB_DIR"
echo "To sync logs: cd $WANDB_DIR && wandb sync ."