#!/bin/bash
#
# ORNL Frontier Supercomputer SLURM Template for Accelerate + Hydra
# 
#SBATCH --output=slurm/hydra_experiment.%j.log
#SBATCH --error=slurm/hydra_experiment.%j.err
#SBATCH --account=mph121                   # Your ORNL project ID
#SBATCH --partition=batch                   # batch is the main partition
#SBATCH --qos=debug                         # debug queue for faster scheduling
#SBATCH --time=02:00:00                     # 2 hours walltime
#SBATCH --job-name=hydra_vae_experiments
#SBATCH --nodes=1                           # Single node for experiments
#SBATCH --mail-type=BEGIN                   # Email when job begins
#SBATCH --mail-user=jmuneton@stanford.edu   # Your email
#SBATCH --ntasks-per-node=8                 # 8 GPUs per node
#SBATCH --cpus-per-task=7                   # 7 CPUs per task (low-noise mode)
#SBATCH --gpus-per-node=8                   # 8 GPUs per node on Frontier
#SBATCH --gpu-bind=closest

# Load required modules for Frontier
module load PrgEnv-amd
module load rocm/5.4.0                      # Adjust version as needed
module load cray-mpich

# Set up environment for AMD GPUs
export MPICH_GPU_SUPPORT_ENABLED=1
export FI_MR_CACHE_MONITOR=memhooks
export FI_CXI_RX_MATCH_MODE=software

# PyTorch/ROCm environment variables
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:512
export HIP_FORCE_DEV_KERNARG=1
export NCCL_DEBUG=INFO
export TORCH_NCCL_BLOCKING_WAIT=0

# Fix "Address family not supported by protocol" errno 97 errors
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_IB_DISABLE=1

# Configure MIOpen cache
export MIOPEN_USER_DB_PATH="/tmp/my-miopen-cache-$USER-$SLURM_JOB_ID"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

# OpenMP settings
export OMP_NUM_THREADS=7

# Create slurm directory if it doesn't exist
mkdir -p slurm

# Experiment configuration
LATENT_CHANNELS=1
NUM_EPOCHS=20
B_VAE=1
KL_MODEL="vae_kl"
VQ_MODEL="vq"
LOSSES=("mse" "l1" "iwmse")
DATASETS=(422 522)

echo "Running experiments for model: $KL_MODEL"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node list: $SLURM_JOB_NODELIST"

# Function to run accelerate command with proper SLURM integration
run_experiment() {
    local model=$1
    local dataset=$2
    local loss=$3
    local batch_size=$4
    local epochs=$5
    local channels=$6
    
    echo "================================================"
    echo "Running: Model=$model, Dataset=$dataset, Loss=$loss"
    echo "Batch Size=$batch_size, Epochs=$epochs, Channels=$channels"
    echo "================================================"
    
    # Use srun with accelerate for proper distributed execution
    srun --ntasks=8 \
         --ntasks-per-node=8 \
         --cpus-per-task=7 \
         --gpus-per-node=8 \
         --gpu-bind=closest \
         accelerate launch \
         --config_file accelerate_config.yaml \
         run_hydra_experiment.py \
         model=$model \
         training.batch_size=$batch_size \
         training.num_epochs=$epochs \
         training.train_vae_from_scratch=true \
         training.test_pipeline=true \
         training.recons_loss=$loss \
         model.latent_channels=$channels \
         data.data_id=$dataset
}

# Run experiments for VAE model
for dataset in "${DATASETS[@]}"; do
    for loss in "${LOSSES[@]}"; do
        run_experiment $KL_MODEL $dataset $loss $B_VAE $NUM_EPOCHS $LATENT_CHANNELS
        
        # Check if job was successful
        if [ $? -ne 0 ]; then
            echo "ERROR: Experiment failed for model=$KL_MODEL, dataset=$dataset, loss=$loss"
            continue
        fi
        
        echo "SUCCESS: Completed experiment for model=$KL_MODEL, dataset=$dataset, loss=$loss"
        echo "Waiting 10 seconds before next experiment..."
        sleep 10
    done
done

echo "All experiments completed!"
echo "Job finished at: $(date)"