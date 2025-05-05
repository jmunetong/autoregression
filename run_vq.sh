#!/bin/bash
TEST_FLAG=$1
LATENT_CHANNELS=1
NUM_EPOCHS=15
B_VAE=4
VQ_MODEL="vq"
LOSSES=("mse" "l1" "iwmse")
DATASETS=(522 422)

echo "Running experiments for model: $VQ_MODEL"
for dataset in "${DATASETS[@]}"; do
    accelerate launch run_experiment.py -m $VQ_MODEL -b $B_VAE --latent_channels $LATENT_CHANNELS --num_epochs $NUM_EPOCHS --data_id $dataset $TEST_FLAG
done

echo "✅ Successfully ran all experiments with no problems."