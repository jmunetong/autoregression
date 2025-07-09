#!/bin/bash
TEST_FLAG=$1
LATENT_CHANNELS=3
NUM_EPOCHS=2
B_VAE=4
VQ_MODEL="vq"
LOSSES=("mse" "l1" "iwmse")
DATASETS=(522 422)


accelerate launch --multi_gpu run_experiment.py --diff --diff_epochs $NUM_EPOCHS --lr "3e-4" --pretrained_vae "experiments/vae_kl/250527-0953_vae_kl_d522_c5502af7" $TEST_FLAG

echo "✅ Successfully ran all experiments with no problems."