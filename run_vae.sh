#!/bin/bash
TEST_FLAG=$1
LATENT_CHANNELS=4
NUM_EPOCHS=12
B_VAE=2
KL_MODEL="vae_kl"
VQ_MODEL="vq"
LOSSES=("iwmse" "mse " "l1")
DATASETS=(522)


echo "Running experiments for model: $KL_MODEL"
for dataset in "${DATASETS[@]}"; do
    for loss in "${LOSSES[@]}"; do
        accelerate launch --multi_gpu run_experiment.py -m $KL_MODEL -b $B_VAE --latent_channels $LATENT_CHANNELS --num_epochs $NUM_EPOCHS --data_id $dataset -rls $loss $TEST_FLAG -ua
        # python run_experiment.py -m $KL_MODEL -b $B_VAE --latent_channels $LATENT_CHANNELS --num_epochs $NUM_EPOCHS --data_id $dataset -rls $loss $TEST_FLAG -ua
    done
done

echo "✅ Successfully ran all experiments with no problems."


# run_experiment.py -m $KL_MODEL -b $B_VAE --latent_channels $LATENT_CHANNELS --num_epochs $NUM_EPOCHS --data_id $dataset -rls $loss $TEST_FLAG -ua