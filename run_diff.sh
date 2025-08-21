#!/bin/bash

# Parse command line arguments
TEST_FLAG=""
AVG_POOLING=""
EPOCHS=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --test)
      TEST_FLAG="--test"
      shift
      ;;
    --avg_pooling)
      AVG_POOLING="--avg_pooling"
      shift
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option $1"
      echo "Usage: $0 [--test] [--avg_pooling] [--epochs NUM_EPOCHS]"
      exit 1
      ;;
  esac
done

# Default values
LATENT_CHANNELS=3
NUM_EPOCHS=${EPOCHS:-2}  # Use provided epochs or default to 2
B_VAE=4
VQ_MODEL="vq"
LOSSES=("mse" "l1" "iwmse")
DATASETS=(522 422)

# Build the command
CMD="accelerate launch --multi_gpu run_experiment.py --diff --diff_epochs $NUM_EPOCHS --lr \"3e-4\" --pretrained_vae \"experiments/vae_kl/250527-0953_vae_kl_d522_c5502af7\""

# Add optional flags
if [[ -n "$TEST_FLAG" ]]; then
    CMD="$CMD $TEST_FLAG"
fi

if [[ -n "$AVG_POOLING" ]]; then
    CMD="$CMD $AVG_POOLING"
fi

# Execute the command
eval $CMD

echo "✅ Successfully ran all experiments with no problems."