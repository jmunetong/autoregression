#!/bin/bash

LATENT_CHANNELS=1
NUM_EPOCHS=20
B_VAE=2
KL_MODEL="vae_kl"
VQ_MODEL="vq"
LOSSES=("mse" "l1" "iwmse")
DATASETS=(422 522)

echo "Running experiments for model: $KL_MODEL"
for dataset in "${DATASETS[@]}"; do
    for loss in "${LOSSES[@]}"; do
        accelerate launch run_experiment.py -m $KL_MODEL -b $B_VAE --latent_channels $LATENT_CHANNELS --num_epochs $NUM_EPOCHS --data_id $dataset -rls $loss 
    done
done

echo "Running experiments for model: $VQ_MODEL"
for dataset in "${DATASETS[@]}"; do
    accelerate launch run_experiment.py -m $VQ_MODEL -b $B_VAE --latent_channels $LATENT_CHANNELS --num_epochs $NUM_EPOCHS --data_id $dataset 
done

echo "✅ Successfully ran all experiments with no problems."