#!/bin/bash

# Parse command line arguments
TEST_FLAG=""
AVG_POOLING=""
EPOCHS=""
CONFIG_OVERRIDES=""

while [[ $# -gt 0 ]]; do
  case $1 in
    -t)
      TEST_FLAG="test_pipeline=true"
      shift
      ;;
    --avg_pooling)
      AVG_POOLING="data.avg_pooling=true"
      shift
      ;;
    --epochs)
      EPOCHS="training.num_epochs=$2"
      shift 2
      ;;
    --config)
      CONFIG_NAME="$2"
      shift 2
      ;;
    *)
      echo "Unknown option $1"
      echo "Usage: $0 [-t] [--avg_pooling] [--epochs NUM_EPOCHS] [--config CONFIG_NAME]"
      exit 1
      ;;
  esac
done

# Build config overrides
OVERRIDES=""
if [[ -n "$TEST_FLAG" ]]; then
    OVERRIDES="$OVERRIDES $TEST_FLAG"
fi

if [[ -n "$AVG_POOLING" ]]; then
    OVERRIDES="$OVERRIDES $AVG_POOLING"
fi

if [[ -n "$EPOCHS" ]]; then
    OVERRIDES="$OVERRIDES $EPOCHS"
fi

# Default values for existing compatibility
DEFAULT_OVERRIDES="diffusion=direct training.lr=3e-4 model.pretrained_vae_path=experiments/vae_kl/250527-0953_vae_kl_d522_c5502af7"

# Build the command
CMD="accelerate launch --multi_gpu run_experiment.py"

# Add config overrides
if [[ -n "$CONFIG_NAME" ]]; then
    CMD="$CMD --config-name=$CONFIG_NAME"
fi

# Add all overrides
CMD="$CMD $DEFAULT_OVERRIDES $OVERRIDES"

# Execute the command
echo "Running: $CMD"
eval $CMD

echo "✅ Successfully ran all experiments with no problems."
