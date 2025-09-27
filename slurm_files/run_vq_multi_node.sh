#!/bin/bash
#SBATCH --account=mph121
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=100
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=8
#SBATCH --job-name=vq_run_522
#SBATCH --output=slurm/vq_522-%j.out
#SBATCH --error=slurm/vq_522-%j.err
#SBATCH --mail-type=END,FAIL  
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

echo "Node: $(hostname)"
echo "SLURM_JOB_NODELIST: '$SLURM_JOB_NODELIST'"
echo "SLURM_NODEID: $SLURM_NODEID"

# Load modules
module load PrgEnv-amd
module load rocm/6.2.4
module load cray-mpich

export MPICH_GPU_SUPPORT_ENABLED=1
export FI_MR_CACHE_MONITOR=memhooks
export FI_CXI_RX_MATCH_MODE=software

# PyTorch/ROCm environment variables
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:512
export HIP_FORCE_DEV_KERNARG=1
export NCCL_DEBUG=INFO
export TORCH_NCCL_BLOCKING_WAIT=0

# Configure MIOpen cache
export MIOPEN_USER_DB_PATH="/tmp/my-miopen-cache-$USER-$SLURM_JOB_ID"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

# CRITICAL: Robust MASTER_ADDR extraction
if [[ -z "$SLURM_JOB_NODELIST" ]]; then
    echo "ERROR: SLURM_JOB_NODELIST is empty!"
    exit 1
fi

# Get the first node from the nodelist and ensure IPv4 resolution
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
if [[ -z "$MASTER_ADDR" ]]; then
    # Fallback method
    MASTER_ADDR=$(echo "$SLURM_JOB_NODELIST" | cut -d',' -f1 | sed 's/\[.*//g')
fi

# CRITICAL: Resolve to IPv4 address to avoid IPv6 issues
MASTER_ADDR_IPv4=$(getent hosts "$MASTER_ADDR" | awk '{print $1}' | head -n 1)
if [[ -n "$MASTER_ADDR_IPv4" ]] && [[ "$MASTER_ADDR_IPv4" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Resolved $MASTER_ADDR to IPv4: $MASTER_ADDR_IPv4"
    MASTER_ADDR="$MASTER_ADDR_IPv4"
else
    echo "Warning: Could not resolve to IPv4, using hostname: $MASTER_ADDR"
fi

# Validate MASTER_ADDR
if [[ -z "$MASTER_ADDR" ]] || [[ "$MASTER_ADDR" == "None" ]]; then
    echo "ERROR: Could not determine MASTER_ADDR!"
    echo "SLURM_JOB_NODELIST: '$SLURM_JOB_NODELIST'"
    echo "Attempting manual extraction..."
    scontrol show hostnames "$SLURM_JOB_NODELIST"
    exit 1
fi

# Network configuration - SET ONCE AND DON'T OVERRIDE
export MASTER_ADDR="$MASTER_ADDR"
# Use a unique port based on job ID to avoid conflicts
export MASTER_PORT=$((23456 + ($SLURM_JOB_ID % 1000)))
export NODE_RANK=$SLURM_NODEID
export LOCAL_RANK=$SLURM_LOCALID
export RANK=$SLURM_PROCID
export WORLD_SIZE=$SLURM_NTASKS

# CRITICAL: Force IPv4 only - fix errno 97
export TORCH_DISTRIBUTED_IPv6=0
export GLOO_SOCKET_IFNAME=hsn0
export NCCL_SOCKET_IFNAME=hsn0
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_FAMILY=AF_INET

# Additional IPv4 enforcement
export NCCL_SOCKET_FORCE_AF_INET=1
export TORCH_NCCL_USE_COMM_NONBLOCKING=0
export TORCH_DISTRIBUTED_USE_IPV6=0

# Gloo backend IPv4 enforcement
export GLOO_SOCKET_FAMILY=AF_INET
export GLOO_DEVICE_TRANSPORT=TCP

# WandB configuration
export WANDB_MODE=offline
export WANDB_DIR="$HOME/wandb_offline_logs"
export WANDB_API_KEY="bbd9552411458167539449b8da39ae4718b9617a"
mkdir -p $HOME/wandb_offline_logs

# Accelerate configuration via environment variables
export ACCELERATE_MIXED_PRECISION="bf16"
export ACCELERATE_USE_CPU="false"
# Specify which accelerate config file to use
export ACCELERATE_CONFIG_FILE="multinode_config.yaml"

# Print debug information
echo "=== SLURM Job Allocation Verification ==="
echo "Requested nodes: 2"
echo "Allocated nodes: $SLURM_NNODES"
echo "Node list: $SLURM_JOB_NODELIST"
echo "All allocated hostnames:"
scontrol show hostnames "$SLURM_JOB_NODELIST"
echo "Total tasks: $SLURM_NTASKS"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "Master port: $MASTER_PORT"
echo "=========================================="

# Test basic network connectivity only
echo "Testing IPv4 connectivity to master node..."
if ping -4 -c 1 "$MASTER_ADDR" &>/dev/null; then
    echo "✓ Can ping master node via IPv4: $MASTER_ADDR"
else
    echo "✗ Cannot ping master node via IPv4: $MASTER_ADDR"
    exit 1
fi

# Verify PyTorch setup
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
print(f'Current device: {torch.cuda.current_device() if torch.cuda.is_available() else \"N/A\"}')
"

# Use srun to launch the training script directly with Python
# Don't use accelerate launch since your Python script uses Accelerator() internally
srun --ntasks=$SLURM_NTASKS \
     --ntasks-per-node=$SLURM_NTASKS_PER_NODE \
     --cpus-per-task=$SLURM_CPUS_PER_TASK \
     bash -c "
export RANK=\$SLURM_PROCID
export LOCAL_RANK=\$SLURM_LOCALID
export WORLD_SIZE=\$SLURM_NTASKS
export MASTER_ADDR='$MASTER_ADDR'
export MASTER_PORT='$MASTER_PORT'
export NODE_RANK=\$SLURM_NODEID

# Calculate machine rank (which node this process is on)
export MACHINE_RANK=\$SLURM_NODEID

# Debug info for each process
echo \"Process \$SLURM_PROCID on node \$(hostname): LOCAL_RANK=\$SLURM_LOCALID, NODE_RANK=\$SLURM_NODEID, MACHINE_RANK=\$MACHINE_RANK\"

# **IMPORTANT**: Use Python directly, not accelerate launch
# Your Python script handles Accelerator() internally
python run_hydra_experiment.py \
    model=vq \
    experiment_type=$EXPERIMENT_TYPE \
    experiment_type.recons_loss=l1 \
    data=full_522
"

echo "Multi-node experiment completed"
echo "Logs saved to: $WANDB_DIR"
echo "To sync logs: cd $WANDB_DIR && wandb sync ."