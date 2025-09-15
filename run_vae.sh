#!/bin/bash
TEST_FLAG=$1
EPOCH_FLAG=
LATENT_CHANNELS=4
NUM_EPOCHS=12
B_VAE=2
KL_MODEL="vae_kl"
VQ_MODEL="vq"
LOSSES=("l1")
DATASETS=(522)


echo "Running experiments for model: $KL_MODEL"
for dataset in "${DATASETS[@]}"; do
    for loss in "${LOSSES[@]}"; do
        # accelerate launch --multi_gpu run_experiment.py -m $KL_MODEL -b $B_VAE --latent_channels $LATENT_CHANNELS --num_epochs $NUM_EPOCHS --data_id $dataset -rls $loss $TEST_FLAG -ua --train_vae_from_scratch

          accelerate launch --multi_gpu run_hydra_experiment.py model=$KL_MODEL  experiment_type=test experiment_type.recons_loss=$loss data.data_id=$dataset
        # python run_experiment.py -m $KL_MODEL -b $B_VAE --latent_channels $LATENT_CHANNELS --num_epochs $NUM_EPOCHS --data_id $dataset -rls $loss $TEST_FLAG -ua
    done
done

echo "✅ Successfully ran all experiments with no problems."

# This is format for formatting the command to run training from scratch
# python run_experiment.py -t -m "vq" --avg_pooling -ua --train_vae_from_scratch
# run_experiment.py -m $KL_MODEL -b $B_VAE --latent_channels $LATENT_CHANNELS --num_epochs $NUM_EPOCHS --data_id $dataset -rls $loss $TEST_FLAG -ua