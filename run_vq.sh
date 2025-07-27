#!/bin/bash
TEST_FLAG=$1
LATENT_CHANNELS=1
NUM_EPOCHS=10
B_VAE=1
VQ_MODEL="vq"
# LOSSES=("iwmse" "mse" "l1")
# DATASETS=(522 422)
DATASETS=(522)

echo "Running experiments for model: $VQ_MODEL"
for dataset in "${DATASETS[@]}"; do
    for loss in "${LOSSES[@]}"; do
    accelerate launch --multi_gpu run_experiment.py -m $VQ_MODEL -b $B_VAE --latent_channels $LATENT_CHANNELS --num_epochs $NUM_EPOCHS --data_id $dataset -rls $loss $TEST_FLAG
    done 
done

echo "✅ Successfully ran all experiments with no problems."