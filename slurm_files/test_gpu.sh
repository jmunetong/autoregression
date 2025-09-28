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
module load PrgEnv-amd
module load rocm/6.2.4
module load cray-mpich
module load miniforge3/23.11.0-0  # CRITICAL: Add this back

# CRITICAL: Properly activate conda environment
source /sw/frontier/spack-envs/base/opt/cray-sles15-zen3/miniforge3-23.11.0-0/etc/profile.d/conda.sh
conda activate pytorch_env

# Verify conda environment is active
echo "=== Environment Check ==="
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"
echo "Python version: $(python --version)"

# Set up environment for AMD GPUs
export ROCM_PATH=/opt/rocm
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/lib64:$LD_LIBRARY_PATH
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# PyTorch/ROCm environment variables
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:512
export HIP_FORCE_DEV_KERNARG=1
export HSA_OVERRIDE_GFX_VERSION=9.0.0

# Distributed training setup
export PYTHONUNBUFFERED=1
export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_DEBUG=INFO
export TORCH_NCCL_BLOCKING_WAIT=0

# MIOpen cache setup
export MIOPEN_USER_DB_PATH="/tmp/my-miopen-cache-$USER-$SLURM_JOB_ID"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

# Network setup
NODE_NAME=$(hostname -s)
export MASTER_ADDR=$NODE_NAME
export MASTER_PORT=29500

# WandB configuration
export WANDB_MODE=offline
export WANDB_DIR="$HOME/wandb_offline_logs"
export WANDB_API_KEY="bbd9552411458167539449b8da39ae4718b9617a"
mkdir -p $HOME/wandb_offline_logs

echo "=== Multi-GPU Configuration ==="
echo "Hostname: $NODE_NAME"
echo "SLURM_GPUS_PER_NODE: $SLURM_GPUS_PER_NODE"
echo "ROCM_PATH: $ROCM_PATH"
echo "HIP_VISIBLE_DEVICES: $HIP_VISIBLE_DEVICES"

# Verify PyTorch and ROCm
python -c "
import torch
print('=== PyTorch ROCm Check ===')
print(f'PyTorch version: {torch.__version__}')
print(f'Built with ROCm: {\"rocm\" in torch.__version__.lower()}')
print(f'HIP version: {getattr(torch.version, \"hip\", \"None\")}')
print(f'ROCm available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'GPU {i} memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB')
else:
    print('❌ No ROCm devices detected!')
    print('Check:')
    print('  1. ROCm PyTorch installation')
    print('  2. Environment variables')
    print('  3. Module loading')
"

# Only proceed if GPUs are detected
if ! python -c "import torch; exit(0 if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 1)"; then
    echo "ERROR: No GPUs detected by PyTorch. Exiting."
    exit 1
fi

# Run with accelerate launch for multi-GPU
ACCELERATE_CONFIG_FILE="accelerate_config/multigpu_config.yaml"

if [[ ! -f "$ACCELERATE_CONFIG_FILE" ]]; then
    echo "ERROR: Accelerate config file not found: $ACCELERATE_CONFIG_FILE"
    exit 1
fi

echo "=== Starting Training ==="
accelerate launch --config_file $ACCELERATE_CONFIG_FILE \
    run_hydra_experiment.py \
    model=vae_kl \
    experiment_type=${EXPERIMENT_TYPE}_iwmse
    data=full

echo "Multi-GPU experiment completed"