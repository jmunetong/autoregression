#!/bin/bash

# Initialize defaults
TEST_FLAG=""
NUM_EPOCHS=14
LATENT_CHANNELS=4
B_VAE=1
VQ_MODEL="vq"
LOSSES=("l1")
DATASETS=(522)
AVG_POOLING_FLAG=""  # Add this line

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--test)
            TEST_FLAG="-t"
            shift
            ;;
        -e|--epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        -a|--avg-pooling)
            AVG_POOLING_FLAG="--avg_pooling"  # Add this case
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -t, --test        Enable test mode"
            echo "  -e, --epochs      Set number of epochs (default: 14)"
            echo "  -a, --avg-pooling Enable average pooling"  # Add this line
            echo "  -h, --help        Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

echo "Running experiments for model: $VQ_MODEL"
echo "Configuration: Epochs=$NUM_EPOCHS, Test mode=$([[ -n "$TEST_FLAG" ]] && echo "enabled" || echo "disabled"), Average pooling=$([[ -n "$AVG_POOLING_FLAG" ]] && echo "enabled" || echo "disabled")"

for dataset in "${DATASETS[@]}"; do
    for loss in "${LOSSES[@]}"; do
    accelerate launch --multi_gpu run_experiment.py -m $VQ_MODEL -b $B_VAE --latent_channels $LATENT_CHANNELS --num_epochs $NUM_EPOCHS --data_id $dataset -rls $loss $TEST_FLAG --train_vae_from_scratch $AVG_POOLING_FLAG
    done 
done

echo "✅ Successfully ran all experiments with no problems."