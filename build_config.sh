#!/bin/bash

# Create directory structure
echo "Creating Hydra configuration directory structure..."

mkdir -p configs/{model,training,data,diffusion,inference,experiment}

# Create main config file
cat > configs/config.yaml << 'EOF'
# configs/config.yaml
defaults:
  - model: vae_kl
  - training: default
  - data: default
  - diffusion: default
  - inference: default
  - _self_

# Global settings
test_pipeline: false
seed: 42

# Hydra configuration
hydra:
  run:
    dir: outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}
EOF

# Create model configs
cat > configs/model/vae_kl.yaml << 'EOF'
# configs/model/vae_kl.yaml
model_name: "vae_kl"
latent_channels: 4
use_annealing: false
annealing_shape: "cosine"  # linear, cosine, logistic

# Training from checkpoint/scratch
train_vae_from_checkpoint: false
train_vae_from_scratch: false
train_diff_from_checkpoint: false
train_diff_from_scratch: false

# Pretrained model paths
pretrained_vae_path: null
pretrained_diff_path: null
EOF

cat > configs/model/vq.yaml << 'EOF'
# configs/model/vq.yaml
model_name: "vq"
latent_channels: 3

# Training from checkpoint/scratch
train_vae_from_checkpoint: false
train_vae_from_scratch: false
train_diff_from_checkpoint: false
train_diff_from_scratch: false

# Pretrained model paths
pretrained_vae_path: null
pretrained_diff_path: null
EOF

# Create training config
cat > configs/training/default.yaml << 'EOF'
# configs/training/default.yaml
batch_size: 3
num_epochs: 20
lr: 1e-4
weight_decay: 1e-3

# Loss parameters
beta_recons: 0.5
recons_loss: "mse"  # mse, l1, iwmse
alpha_mse: 2.0

# EMA parameters
ema_decay: 0.9999
EOF

# Create data configs
cat > configs/data/default.yaml << 'EOF'
# configs/data/default.yaml
data_id: 522  # 422, 522
avg_pooling: false
topk: 1.0
data_path: ${oc.env:DATA_PATH}  # Uses environment variable
train_ratio: 0.8
seed: 42
EOF

cat > configs/data/experiment_422.yaml << 'EOF'
# configs/data/experiment_422.yaml
data_id: 422
avg_pooling: false
topk: 1.0
data_path: ${oc.env:DATA_PATH}
train_ratio: 0.8
seed: 42
EOF

cat > configs/data/pooled.yaml << 'EOF'
# configs/data/pooled.yaml
data_id: 522
avg_pooling: true
topk: 1.0
data_path: ${oc.env:DATA_PATH}
train_ratio: 0.8
seed: 42
EOF

# Create diffusion configs
cat > configs/diffusion/default.yaml << 'EOF'
# configs/diffusion/default.yaml
diff: false
latent_diff: false
diff_epochs: 10
patch_size: 16
vit_size: "base"  # base, large, huge
EOF

cat > configs/diffusion/latent.yaml << 'EOF'
# configs/diffusion/latent.yaml
diff: false
latent_diff: true
diff_epochs: 10
patch_size: 16
vit_size: "base"
EOF

cat > configs/diffusion/direct.yaml << 'EOF'
# configs/diffusion/direct.yaml
diff: true
latent_diff: false
diff_epochs: 10
patch_size: 16
vit_size: "base"
EOF

# Create inference config
cat > configs/inference/default.yaml << 'EOF'
# configs/inference/default.yaml
inference: false
generate_samples: false
num_samples: 10
EOF

# Create experiment configs
cat > configs/experiment/vae_training.yaml << 'EOF'
# configs/experiment/vae_training.yaml
# @package _global_
defaults:
  - override /model: vae_kl
  - override /training: default
  - override /data: default
  - override /diffusion: default

model:
  train_vae_from_scratch: true

training:
  num_epochs: 50
  batch_size: 8
EOF

cat > configs/experiment/latent_diffusion.yaml << 'EOF'
# configs/experiment/latent_diffusion.yaml
# @package _global_
defaults:
  - override /model: vae_kl
  - override /training: default
  - override /data: default
  - override /diffusion: latent

model:
  pretrained_vae_path: "experiments/vae_kl/250527-0953_vae_kl_d522_c5502af7"
  train_diff_from_scratch: true

diffusion:
  diff_epochs: 20

training:
  lr: 3e-4
EOF

cat > configs/experiment/direct_diffusion.yaml << 'EOF'
# configs/experiment/direct_diffusion.yaml
# @package _global_
defaults:
  - override /diffusion: direct
  - override /training: default
  - override /data: default

diffusion:
  diff_epochs: 30
  patch_size: 16

model:
  train_diff_from_scratch: true

training:
  lr: 1e-4
  batch_size: 4
EOF

cat > configs/experiment/inference_only.yaml << 'EOF'
# configs/experiment/inference_only.yaml
# @package _global_
defaults:
  - override /inference: default
  - override /diffusion: latent

inference:
  inference: true
  generate_samples: true
  num_samples: 50

model:
  pretrained_vae_path: "path/to/vae/model"
  pretrained_diff_path: "path/to/diff/model"
EOF

# Create updated bash script
cat > run_hydra_experiment.sh << 'EOF'
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
EOF

# Make bash script executable
chmod +x run_hydra_experiment.sh

echo "✅ Hydra configuration structure created successfully!"
echo ""
echo "Directory structure:"
tree configs/ 2>/dev/null || find configs -type f | sort

echo ""
echo "Files created:"
echo "- configs/ (directory with all config files)"
echo "- run_hydra_experiment.sh (updated bash script)"
echo "- run_experiment.py (you'll need to replace your existing file)"

echo ""
echo "Next steps:"
echo "1. Replace your existing run_experiment.py with the Hydra version"
echo "2. Install hydra: pip install hydra-core"
echo "3. Test with: python run_experiment.py --help"
echo "4. Run experiments: ./run_hydra_experiment.sh -t --avg_pooling --epochs 25"
EOF