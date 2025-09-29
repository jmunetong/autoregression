# Scaling X-ray Diffraction Modeling with Accelerate, Hydra, and SLURM: A Complete Guide to Multi-GPU VAE, VQ, and Diffusion Models

Large-scale X-ray diffraction data modeling requires sophisticated computational infrastructure to train complex neural networks like VAEs, VQ-VAEs, and diffusion models. In this comprehensive guide, we'll demonstrate how to leverage three powerful tools—Accelerate, Hydra, and SLURM—to seamlessly scale your X-ray diffraction experiments across multiple GPUs and compute nodes.

## Overview: The Power Trio for Scalable Scientific Computing

### Why This Stack Works for X-ray Diffraction

X-ray diffraction data presents unique challenges:
- **High-dimensional data**: Diffraction patterns often require complex neural architectures
- **Large datasets**: Synchrotron facilities generate massive datasets requiring distributed training
- **Model complexity**: VAEs, VQ-VAEs, and diffusion models demand significant computational resources
- **Experiment tracking**: Scientific workflows need systematic hyperparameter management

Our stack addresses these challenges:
- **Accelerate**: Seamless multi-GPU/multi-node training without code changes
- **Hydra**: Elegant configuration management for complex experimental workflows
- **SLURM**: Enterprise-grade job scheduling for HPC environments

## Mathematical Foundations: Understanding VAEs and VQ-VAEs for X-ray Diffraction

Before diving into the implementation details, let's establish the mathematical foundations of the models we'll be training. Understanding these concepts is crucial for effective hyperparameter tuning and interpreting results in X-ray diffraction analysis.

### Variational Autoencoders (VAEs): Theory and Application to Diffraction Data

#### Core Mathematical Framework

A Variational Autoencoder learns to encode X-ray diffraction patterns $\mathbf{x} \in \mathbb{R}^D$ into a lower-dimensional latent space $\mathbf{z} \in \mathbb{R}^d$ (where $d \ll D$), while ensuring the latent space follows a meaningful probability distribution.

**The VAE Objective Function:**

The VAE optimizes the Evidence Lower BOund (ELBO):

$$\mathcal{L}_{\text{VAE}} = \mathbb{E}_{q_{\phi}(\mathbf{z}|\mathbf{x})}[\log p_{\theta}(\mathbf{x}|\mathbf{z})] - \beta \cdot D_{KL}(q_{\phi}(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$$

Where:
- **Reconstruction Term**: $\mathbb{E}_{q_{\phi}(\mathbf{z}|\mathbf{x})}[\log p_{\theta}(\mathbf{x}|\mathbf{z})]$ ensures decoded patterns match input diffraction data
- **Regularization Term**: $D_{KL}(q_{\phi}(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$ keeps latent distributions close to a prior $p(\mathbf{z}) = \mathcal{N}(0, I)$
- **β**: Weighting factor controlling reconstruction vs. regularization trade-off

#### VAE Components for X-ray Diffraction

**1. Encoder Network (Recognition Model):**
$$q_{\phi}(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}_{\phi}(\mathbf{x}), \boldsymbol{\sigma}^2_{\phi}(\mathbf{x})I)$$

- Maps diffraction pattern $\mathbf{x}$ to latent parameters $\boldsymbol{\mu}_{\phi}(\mathbf{x})$ and $\boldsymbol{\sigma}_{\phi}(\mathbf{x})$
- $\mathbf{x}$: Input diffraction pattern (e.g., 512×512 intensity map)
- $\boldsymbol{\mu}_{\phi}(\mathbf{x})$: Mean of latent distribution (learned function of input)
- $\boldsymbol{\sigma}_{\phi}(\mathbf{x})$: Standard deviation of latent distribution

**2. Reparameterization Trick:**
$$\mathbf{z} = \boldsymbol{\mu}_{\phi}(\mathbf{x}) + \boldsymbol{\sigma}_{\phi}(\mathbf{x}) \odot \boldsymbol{\epsilon}, \quad \text{where } \boldsymbol{\epsilon} \sim \mathcal{N}(0, I)$$

This enables backpropagation through the stochastic sampling process.

**3. Decoder Network (Generative Model):**
$$p_{\theta}(\mathbf{x}|\mathbf{z}) = \mathcal{N}(\boldsymbol{\mu}_{\theta}(\mathbf{z}), \sigma^2 I) \text{ for continuous intensities}$$

- Reconstructs diffraction pattern from latent code $\mathbf{z}$
- For X-ray data: often use Gaussian likelihood for continuous intensities

#### Loss Functions for X-ray Diffraction

The framework implements three reconstruction losses, each suited for different diffraction characteristics:

**1. Mean Squared Error (MSE):**
$$\mathcal{L}_{\text{recon}}^{\text{MSE}} = \|\mathbf{x} - \hat{\mathbf{x}}\|_2^2 = \sum_i (x_i - \hat{x}_i)^2$$

- **Best for**: Smooth diffraction patterns, Gaussian noise
- **Characteristics**: Tends to blur sharp diffraction peaks

**2. L1 Loss (Manhattan Distance):**
$$\mathcal{L}_{\text{recon}}^{\text{L1}} = \|\mathbf{x} - \hat{\mathbf{x}}\|_1 = \sum_i |x_i - \hat{x}_i|$$

- **Best for**: Sparse diffraction peaks, preserving sharp features
- **Characteristics**: More robust to outliers, maintains peak sharpness

**3. Importance-Weighted MSE (IWMSE):**
$$\mathcal{L}_{\text{recon}}^{\text{IWMSE}} = \sum_i w_i(x_i - \hat{x}_i)^2, \quad \text{where } w_i = f(\text{intensity}_i)$$

- **Best for**: Varying peak intensities, focusing on strong reflections
- **Characteristics**: Weights loss by diffraction peak importance

#### β-VAE for Controllable Latent Representations

The β parameter in the VAE objective controls the disentanglement of latent factors:

$$\mathcal{L}_{\beta\text{-VAE}} = \mathcal{L}_{\text{recon}} + \beta \cdot D_{KL}(q_{\phi}(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$$

**For X-ray Diffraction Applications:**
- **β < 1**: Emphasizes reconstruction quality (preserves fine diffraction details)
- **β = 1**: Standard VAE formulation
- **β > 1**: Promotes disentangled latent factors (separates crystal structure components)

#### KL Divergence Computation

For the Gaussian encoder $q_{\phi}(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma}^2 I)$ and prior $p(\mathbf{z}) = \mathcal{N}(0, I)$:

$$D_{KL}(q_{\phi}(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z})) = \frac{1}{2} \sum_j [\mu_j^2 + \sigma_j^2 - \log(\sigma_j^2) - 1]$$

This closed-form solution makes VAE training efficient and stable.

### Vector Quantized VAEs (VQ-VAEs): Discrete Latent Representations

#### Mathematical Foundation

VQ-VAE replaces the continuous latent space of standard VAEs with a discrete codebook, making it particularly suitable for X-ray diffraction data where crystallographic symmetries are naturally discrete.

**The VQ-VAE Objective:**

$$\mathcal{L}_{\text{VQ-VAE}} = \mathcal{L}_{\text{recon}} + \|\text{sg}[\mathbf{z}_e] - \mathbf{e}\|_2^2 + \beta\|\mathbf{z}_e - \text{sg}[\mathbf{e}]\|_2^2$$

Where:
- **$\mathcal{L}_{\text{recon}}$**: Reconstruction loss (MSE, L1, or IWMSE)
- **Codebook Loss**: $\|\text{sg}[\mathbf{z}_e] - \mathbf{e}\|_2^2$ updates codebook vectors
- **Commitment Loss**: $\beta\|\mathbf{z}_e - \text{sg}[\mathbf{e}]\|_2^2$ encourages encoder commitment
- **$\text{sg}[\cdot]$**: Stop-gradient operator

#### VQ-VAE Components

**1. Encoder Network:**
$$\mathbf{z}_e = \text{Encoder}_{\phi}(\mathbf{x}) \in \mathbb{R}^{H \times W \times D}$$
- Maps diffraction pattern to continuous embeddings
- **$\mathbf{x}$**: Input diffraction pattern
- **$\mathbf{z}_e$**: Encoder output (continuous, pre-quantization)

**2. Vector Quantization:**
$$\mathbf{z}_q(i,j) = \arg\min_k \|\mathbf{z}_e(i,j) - \mathbf{e}_k\|_2$$
- **$\mathbf{E} = \{\mathbf{e}_k\}_{k=1}^K$**: Learned codebook with K vectors
- Each spatial location $(i,j)$ gets assigned to nearest codebook vector
- **$\mathbf{z}_q$**: Quantized representation (discrete)

**3. Straight-Through Estimator:**
$$\mathbf{z}_q = \mathbf{z}_e + \text{sg}[q(\mathbf{z}_e) - \mathbf{z}_e]$$
This allows gradients to flow through the discrete quantization step during backpropagation.

**4. Decoder Network:**
$$\hat{\mathbf{x}} = \text{Decoder}_{\theta}(\mathbf{z}_q)$$

#### VQ-VAE Loss Components Detailed

**1. Reconstruction Loss:**
Same as VAE (MSE, L1, or IWMSE), measuring quality of diffraction pattern reconstruction.

**2. Codebook Loss (Vector Quantization):**
$$\mathcal{L}_{\text{vq}} = \|\text{sg}[\mathbf{z}_e] - \mathbf{e}\|_2^2$$
- Updates codebook vectors $\mathbf{e}$ to minimize distance to encoder outputs
- Stop gradient on $\mathbf{z}_e$ prevents encoder from changing to minimize this loss

**3. Commitment Loss:**
$$\mathcal{L}_{\text{commit}} = \beta\|\mathbf{z}_e - \text{sg}[\mathbf{e}]\|_2^2$$
- Encourages encoder outputs to stay close to chosen codebook vectors
- $\beta$ typically set to 0.25 in practice
- Stop gradient on $\mathbf{e}$ focuses loss on encoder training

#### Codebook Learning Dynamics

The codebook vectors are updated using exponential moving averages:

$$N_i^{(t)} = \lambda \cdot N_i^{(t-1)} + (1-\lambda) \cdot n_i^{(t)}$$
$$m_i^{(t)} = \lambda \cdot m_i^{(t-1)} + (1-\lambda) \cdot \sum_j \mathbf{z}_{e,j}^{(t)}$$
$$\mathbf{e}_i^{(t)} = \frac{m_i^{(t)}}{N_i^{(t)}}$$

Where:
- **$N_i$**: Count of how often codebook vector $i$ is used
- **$m_i$**: Sum of encoder outputs assigned to codebook vector $i$
- **$\lambda$**: Decay factor (typically 0.99)

#### Benefits for X-ray Diffraction

**1. Discrete Crystallographic Representations:**
- Natural for representing distinct crystal structures
- Codebook vectors can learn common diffraction motifs
- Better for classification and structure identification

**2. Sharp Feature Preservation:**
- No posterior collapse (common VAE problem)
- Maintains sharp diffraction peaks better than continuous VAEs
- Ideal for preserving Bragg peak structure

**3. Interpretable Latent Space:**
- Each codebook vector represents a learnable diffraction "building block"
- Enables analysis of structural components in crystallographic data
- Facilitates crystal structure decomposition

#### Comparison: VAE vs VQ-VAE for X-ray Diffraction

| Aspect | VAE | VQ-VAE |
|--------|-----|---------|
| **Latent Space** | Continuous, Gaussian | Discrete, learned codebook |
| **Peak Preservation** | May blur sharp peaks | Excellent sharp feature preservation |
| **Training Stability** | Generally stable | Requires careful β tuning |
| **Crystallographic Interpretation** | Smooth variations | Discrete structural components |
| **Generation Diversity** | High (continuous sampling) | Controlled by codebook size |
| **Memory Usage** | Lower | Higher (stores codebook) |
| **Best Use Cases** | Smooth patterns, interpolation | Sharp peaks, classification |

### Diffusion Models: Probabilistic Denoising for High-Quality Generation

#### Mathematical Foundation

Diffusion models represent a paradigm shift in generative modeling, inspired by non-equilibrium thermodynamics. Instead of learning a direct mapping from noise to data (like VAEs) or adversarial training (like GANs), diffusion models learn to gradually transform pure noise into meaningful data through a carefully designed denoising process.

**Core Principle:**
The key insight is to model data generation as the reverse of a corruption process. We systematically add noise to data, then learn to reverse this process step by step.

#### Forward Diffusion Process (Data Corruption)

The forward process transforms clean X-ray diffraction patterns into pure Gaussian noise through a fixed Markov chain:

$$q(\mathbf{x}_{1:T}|\mathbf{x}_0) = \prod_{t=1}^T q(\mathbf{x}_t|\mathbf{x}_{t-1})$$

$$q(\mathbf{x}_t|\mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t}\mathbf{x}_{t-1}, \beta_t \mathbf{I})$$

Where:
- **$\mathbf{x}_0$**: Original diffraction pattern
- **$\mathbf{x}_t$**: Noisy version at timestep $t$
- **$\beta_t$**: Noise schedule controlling corruption rate
- **$T$**: Total number of diffusion steps (typically 1000)

**Closed-Form Forward Sampling:**
Using the reparameterization trick, we can sample any intermediate state directly:

$$\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$$

Where:
- **$\alpha_t = 1 - \beta_t$**
- **$\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$** (cumulative product)
- **$\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$**: Random noise

This allows efficient training without simulating the entire forward chain.

#### Reverse Diffusion Process (Denoising Generation)

The reverse process learns to undo the corruption, transforming noise back into meaningful diffraction patterns:

$$p_{\theta}(\mathbf{x}_{0:T}) = p(\mathbf{x}_T) \prod_{t=1}^T p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_t)$$

$$p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_{\theta}(\mathbf{x}_t, t), \boldsymbol{\Sigma}_{\theta}(\mathbf{x}_t, t))$$

**Neural Network Parameterization:**
Instead of directly predicting $\boldsymbol{\mu}_{\theta}$, it's more effective to predict the noise:

$$\boldsymbol{\epsilon}_{\theta}(\mathbf{x}_t, t) \approx \boldsymbol{\epsilon}$$

The mean is then computed as:
$$\boldsymbol{\mu}_{\theta}(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\boldsymbol{\epsilon}_{\theta}(\mathbf{x}_t, t)\right)$$

#### Training Objective

**Simplified Loss Function:**
The diffusion model training reduces to a simple denoising objective:

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t,\mathbf{x}_0,\boldsymbol{\epsilon}}\left[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_{\theta}(\mathbf{x}_t, t)\|^2\right]$$

Where:
- **$t \sim \text{Uniform}(1, T)$**: Random timestep
- **$\mathbf{x}_0 \sim q(\mathbf{x}_0)$**: Clean diffraction pattern from dataset
- **$\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$**: Random noise
- **$\mathbf{x}_t$**: Noisy version computed using forward process

**Training Algorithm:**
1. Sample a clean diffraction pattern $\mathbf{x}_0$
2. Sample a random timestep $t$
3. Sample noise $\boldsymbol{\epsilon}$ and create $\mathbf{x}_t$
4. Train network $\boldsymbol{\epsilon}_{\theta}$ to predict the noise
5. Compute loss and backpropagate

#### Noise Scheduling for X-ray Diffraction

The noise schedule $\beta_t$ is crucial for diffusion quality. For diffraction patterns with sharp peaks and varying intensities:

**Linear Schedule:**
$$\beta_t = \beta_1 + \frac{t-1}{T-1}(\beta_T - \beta_1)$$

**Cosine Schedule (often better for images):**
$$\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)^2$$

**For X-ray Diffraction:**
- **$\beta_{1} = 1 \times 10^{-4}$**: Small initial noise preserves fine details
- **$\beta_{T} = 0.02$**: Complete corruption at final step
- **$T = 1000$**: Standard number of diffusion steps

#### Sampling Process (Generation)

To generate new diffraction patterns, we reverse the diffusion process:

**DDPM Sampling:**
```
1. Start with pure noise: x_T ~ N(0, I)
2. For t = T, T-1, ..., 1:
   a. Predict noise: ε_θ(x_t, t)
   b. Compute mean: μ_θ(x_t, t)
   c. Sample: x_{t-1} ~ N(μ_θ(x_t, t), β_t I)
3. Return x_0 (generated diffraction pattern)
```

**Faster Sampling (DDIM):**
DDIM enables deterministic sampling with fewer steps (e.g., 50 instead of 1000):

$$\mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\left(\frac{\mathbf{x}_t - \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}_{\theta}(\mathbf{x}_t, t)}{\sqrt{\bar{\alpha}_t}}\right) + \sqrt{1-\bar{\alpha}_{t-1}}\boldsymbol{\epsilon}_{\theta}(\mathbf{x}_t, t)$$

#### Benefits for X-ray Diffraction

**1. High-Quality Generation:**
- No mode collapse (unlike GANs)
- Captures fine diffraction peak structures
- Handles multi-modal crystal distributions

**2. Stable Training:**
- No adversarial training dynamics
- Robust to hyperparameter choices
- Scales well with model size

**3. Controllable Generation:**
- Classifier guidance for conditional generation
- Inpainting capabilities for incomplete patterns
- Interpolation between different crystal structures

#### Autoregressive Diffusion Framework

Recent advances combine diffusion with autoregressive modeling for enhanced control over generation. Instead of generating entire diffraction patterns at once, **Autoregressive Diffusion** generates patterns sequentially, token by token or region by region.

**Key Innovation (Li et al. 2024):**
Traditional autoregressive models use discrete tokens and categorical cross-entropy loss. The new approach uses a **Diffusion Loss function** to enable autoregressive modeling in continuous space:

$$\mathcal{L}_{\text{AR-Diff}} = \sum_{i=1}^{N} \mathbb{E}_{t,\boldsymbol{\epsilon}}\left[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_{\theta}(\mathbf{x}_{t,i}|\mathbf{x}_{<i}, t)\|^2\right]$$

Where:
- **$\mathbf{x}_{<i}$**: Previously generated tokens/patches
- **$\mathbf{x}_{t,i}$**: Noisy version of current token at timestep $t$
- **$N$**: Total number of tokens/patches

**Benefits for X-ray Diffraction:**
- **Structured Generation**: Respects crystallographic relationships between different regions
- **Conditional Control**: Can generate specific diffraction features given partial patterns
- **Memory Efficiency**: Generates high-resolution patterns incrementally
- **Physical Constraints**: Easier to enforce symmetry and intensity constraints

#### Latent Diffusion: Combining VAE and Diffusion Efficiency

**The Latent Diffusion Alternative:**
Instead of applying diffusion directly to high-resolution diffraction patterns, we can combine the representational power of VAEs/VQ-VAEs with diffusion's generation quality:

**Two-Stage Process:**
1. **Stage 1**: Train a VAE or VQ-VAE to learn a compact latent representation of diffraction patterns
2. **Stage 2**: Train a diffusion model in the learned latent space

**Mathematical Formulation:**
$$\mathbf{z}_0 = \text{Encoder}(\mathbf{x}_0) \quad \text{(VAE/VQ-VAE encoding)}$$
$$\mathbf{z}_t = \sqrt{\bar{\alpha}_t}\mathbf{z}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon} \quad \text{(Latent diffusion)}$$
$$\hat{\mathbf{x}}_0 = \text{Decoder}(\hat{\mathbf{z}}_0) \quad \text{(Reconstruction)}$$

**Advantages for X-ray Diffraction:**

**1. Computational Efficiency:**
- Diffusion operates in compressed latent space (e.g., 64×64 instead of 512×512)
- ~8x faster training and inference
- Reduced memory requirements for high-resolution patterns

**2. Quality Benefits:**
- Combines VAE's reconstruction fidelity with diffusion's generation diversity
- Better handling of sharp diffraction peaks through VAE preprocessing
- Maintains fine crystallographic details

**3. Practical Implementation:**
- Can leverage pre-trained VAE/VQ-VAE models
- Modular training: optimize representation and generation separately
- Easier to incorporate domain-specific constraints in latent space

**Latent Diffusion Training:**
```python
# Pseudo-code for latent diffusion training
for diffraction_pattern in dataloader:
    # Encode to latent space
    z_0 = vae.encode(diffraction_pattern)
    
    # Sample timestep and noise
    t = random.randint(1, T)
    epsilon = torch.randn_like(z_0)
    
    # Add noise in latent space
    z_t = sqrt(alpha_bar_t) * z_0 + sqrt(1 - alpha_bar_t) * epsilon
    
    # Predict noise in latent space
    epsilon_pred = diffusion_model(z_t, t)
    
    # Compute loss
    loss = F.mse_loss(epsilon, epsilon_pred)
```

#### Comparison: Direct vs. Latent Diffusion for X-ray Diffraction

| Aspect | Direct Diffusion | Latent Diffusion |
|--------|------------------|------------------|
| **Training Speed** | Slow (high-res operations) | Fast (latent operations) |
| **Memory Usage** | High | Low |
| **Generation Quality** | Excellent detail preservation | Good quality + efficiency |
| **Fine-tuning** | Requires full retraining | Can fine-tune components separately |
| **Computational Cost** | ~8x higher | Baseline |
| **Best Use Cases** | Maximum quality, unlimited compute | Production systems, large-scale |

## Project Structure

```
autoregression/
├── configs/                     # Hydra configuration hierarchy
│   ├── config.yaml             # Main configuration file
│   ├── model/                  # Model-specific configs
│   │   ├── vae_kl.yaml        # VAE with KL divergence
│   │   ├── vq.yaml            # Vector Quantized VAE
│   │   ├── diff.yaml          # Direct diffusion model
│   │   └── latent_diff.yaml   # Latent diffusion model
│   ├── experiment_type/        # Training configurations
│   │   ├── train.yaml         # Standard training setup
│   │   ├── train_mse.yaml     # MSE loss experiment
│   │   └── train_l1.yaml      # L1 loss experiment
│   └── data/                   # Dataset configurations
│       ├── full.yaml          # Full dataset
│       └── pooling.yaml       # Pooled dataset
├── accelerate_config/          # Accelerate configurations
│   ├── singlegpu_config.yaml  # Single GPU setup
│   ├── multigpu_config.yaml   # Multi-GPU single node
│   └── multinode_config.yaml  # Multi-node setup
├── slurm_files/               # SLURM job scripts
│   ├── run_diff_multi_node.sh # Diffusion multi-node training
│   └── run_vae_multi_node.sh  # VAE multi-node training
├── run_hydra_experiment.py    # Main training script
└── frontier.sbatch           # ORNL Frontier job template
```

## Part 1: Configuring Accelerate for Scientific Workloads

### 1.1 Single GPU Configuration

For development and small-scale experiments:

```yaml
# accelerate_config/singlegpu_config.yaml
compute_environment: LOCAL_MACHINE
distributed_type: NO
downcast_bf16: 'no'
gpu_ids: '0'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

### 1.2 Multi-GPU Single Node Configuration

Perfect for high-end workstations with multiple GPUs:

```yaml
# accelerate_config/multigpu_config.yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: bf16  # Optimized for AMD MI250X GPUs
num_machines: 1
num_processes: 8       # 8 GPUs per node
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

### 1.3 Multi-Node Configuration for HPC

For large-scale experiments on supercomputers like ORNL Frontier:

```yaml
# accelerate_config/multinode_config.yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0  # Overridden by SLURM environment
main_process_ip: localhost  # Overridden by MASTER_ADDR
main_process_port: 23456
main_training_function: main
mixed_precision: bf16
num_machines: 10
num_processes: 80    # 10 nodes × 8 GPUs per node
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

### 1.4 Understanding Multi-GPU and Multi-Node Training: All-Reduce and Gradient Synchronization

#### Single GPU vs. Multi-GPU vs. Multi-Node: Key Differences

**Single GPU Training:**
- Model parameters reside on one GPU
- Gradient computation is straightforward: one backward pass per batch
- No communication overhead
- Limited by single GPU memory and compute capacity

**Multi-GPU Single Node Training:**
- Model is replicated across multiple GPUs on the same machine
- Each GPU processes a different subset of the batch (data parallelism)
- High-bandwidth inter-GPU communication (NVLink/Infinity Fabric)
- Shared system memory and storage

**Multi-Node GPU Training:**
- Model replication across GPUs spanning multiple physical machines
- Network communication between nodes (typically InfiniBand or Ethernet)
- Higher latency communication but massive scalability
- Independent node resources (memory, storage, CPUs)

#### How Gradient Updates Work in Distributed Training

The core challenge in distributed training is ensuring all model replicas stay synchronized. This is achieved through **gradient all-reduce operations** that compute the average gradient across all participating GPUs.

**Mathematical Foundation:**

For a model with parameters $\theta$ trained across $N$ GPUs, each processing a different mini-batch $B_i$:

1. **Forward Pass (Independent):**
   Each GPU $i$ computes its loss independently:
   $$L_i = \frac{1}{|B_i|} \sum_{x \in B_i} \ell(f_{\theta}(x), y)$$

2. **Backward Pass (Independent):**
   Each GPU computes gradients for its mini-batch:
   $$g_i = \nabla_{\theta} L_i$$

3. **All-Reduce Communication:**
   All GPUs participate in computing the average gradient:
   $$\bar{g} = \frac{1}{N} \sum_{i=1}^N g_i$$

4. **Parameter Update (Synchronized):**
   All GPUs update their parameters with the same averaged gradient:
   $$\theta_{new} = \theta_{old} - \eta \bar{g}$$

**The All-Reduce Algorithm:**

All-reduce is efficiently implemented using a ring-based or tree-based approach:

```
Ring All-Reduce Example (4 GPUs):
Initial: GPU0=[g0], GPU1=[g1], GPU2=[g2], GPU3=[g3]

Step 1 - Reduce-Scatter:
GPU0 sends g0 to GPU1, receives g3 from GPU3
GPU1 sends g1 to GPU2, receives g0 from GPU0
... (ring communication)

Step 2 - All-Gather:
Each GPU has partial sums, now broadcast complete average
Final: All GPUs have [ḡ0, ḡ1, ḡ2, ḡ3] where ḡi = (gi_0 + gi_1 + gi_2 + gi_3)/4
```

**Communication Complexity:**
- **Ring All-Reduce**: O(P-1) communication steps for P processes
- **Bandwidth Optimal**: Each GPU sends/receives exactly 2(P-1)/P of total data
- **Latency**: O(P) for ring topology, O(log P) for tree topology

#### Practical Implementation in X-ray Diffraction Training

**Data Distribution Strategy:**

```python
# Each GPU processes different diffraction patterns
def create_distributed_dataloader(dataset, batch_size, num_replicas, rank):
    """
    Creates a distributed sampler ensuring each GPU sees different data
    """
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, 
        num_replicas=num_replicas,  # Total number of GPUs
        rank=rank,                  # Current GPU rank
        shuffle=True
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )

# Usage across 8 GPUs:
# GPU 0: processes samples [0, 8, 16, 24, ...]
# GPU 1: processes samples [1, 9, 17, 25, ...]
# GPU 2: processes samples [2, 10, 18, 26, ...]
# ...
```

**Gradient All-Reduce in Action:**

```python
# Simplified training loop showing gradient synchronization
def training_step_distributed(model, batch, optimizer, accelerator):
    """
    Single training step with automatic gradient all-reduce
    """
    # 1. Forward pass (independent on each GPU)
    with accelerator.autocast():  # Mixed precision
        diffraction_pattern = batch['image']
        
        if model_type == 'vae':
            reconstructed, mu, logvar = model(diffraction_pattern)
            loss = vae_loss(reconstructed, diffraction_pattern, mu, logvar)
        elif model_type == 'diffusion':
            noise = torch.randn_like(diffraction_pattern)
            timesteps = torch.randint(0, 1000, (diffraction_pattern.shape[0],))
            predicted_noise = model(diffraction_pattern, timesteps, noise)
            loss = F.mse_loss(predicted_noise, noise)
    
    # 2. Backward pass (independent gradient computation)
    accelerator.backward(loss)
    
    # 3. All-reduce happens automatically in optimizer.step()
    # Behind the scenes:
    # - Each GPU has computed gradients for its mini-batch
    # - All-reduce operation averages gradients across all GPUs
    # - All GPUs receive the same averaged gradients
    # - Parameters are updated identically across all GPUs
    
    optimizer.step()
    optimizer.zero_grad()
    
    return loss

# The magic: accelerator.prepare() wraps model/optimizer for automatic all-reduce
model, optimizer = accelerator.prepare(model, optimizer)
```

**Communication Patterns for Different Scenarios:**

```python
# Multi-GPU Single Node (8 GPUs)
# Communication: High-bandwidth GPU-to-GPU (NVLink: ~600 GB/s)
# All-reduce latency: ~100-500 microseconds
# Effective for batch_size >= 8 per GPU

# Multi-Node Training (10 nodes × 8 GPUs = 80 GPUs)
# Communication: Inter-node network (InfiniBand: ~25-100 GB/s)
# All-reduce latency: ~1-10 milliseconds  
# Requires larger batch_size >= 16 per GPU to hide communication cost
```

#### Performance Optimization for X-ray Diffraction Models

**Gradient Accumulation for Memory-Constrained Training:**

```python
def gradient_accumulation_training(model, dataloader, optimizer, 
                                 accumulation_steps=4):
    """
    Simulate larger batch sizes through gradient accumulation
    Useful when diffraction patterns are high-resolution and memory-limited
    """
    model.train()
    
    for batch_idx, batch in enumerate(dataloader):
        # Forward pass with scaled loss
        with accelerator.autocast():
            loss = compute_loss(model, batch) / accumulation_steps
        
        # Backward pass (gradients accumulate)
        accelerator.backward(loss)
        
        # All-reduce only every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()  # All-reduce happens here
            optimizer.zero_grad()
            
    # Effective batch size = actual_batch_size × accumulation_steps × num_gpus
    # Memory usage = actual_batch_size per GPU
```

**Overlapping Computation and Communication:**

```python
# Advanced: Manual gradient bucketing for optimal overlap
# Useful for very large diffusion models with millions of parameters

def setup_gradient_bucketing(model, bucket_size_mb=25):
    """
    Groups model parameters into buckets for overlapped all-reduce
    """
    # Accelerate automatically does this, but manual control available:
    from accelerate.utils import DistributedDataParallelKwargs
    
    ddp_kwargs = DistributedDataParallelKwargs(
        bucket_cap_mb=bucket_size_mb,
        find_unused_parameters=False,  # Optimization for static graphs
        gradient_as_bucket_view=True   # Memory optimization
    )
    
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    return accelerator
```

#### Communication Analysis for X-ray Diffraction Workloads

**Scaling Laws and Communication Overhead:**

```python
# Communication cost analysis for different model types
def analyze_communication_cost(model_type, num_params, num_gpus):
    """
    Estimates communication overhead for different configurations
    """
    # Model parameter counts (approximate)
    param_counts = {
        'vae_kl': 50_000_000,      # ~50M parameters
        'vae_kl_8': 80_000_000,    # ~80M parameters  
        'vq': 45_000_000,          # ~45M parameters
        'diff': 200_000_000,       # ~200M parameters (Vision Transformer)
        'latent_diff': 150_000_000  # ~150M parameters
    }
    
    params = param_counts[model_type]
    
    # Communication volume per all-reduce (FP16)
    bytes_per_param = 2  # FP16
    total_bytes = params * bytes_per_param
    
    # Network bandwidth (Frontier InfiniBand)
    bandwidth_gbps = 25  # 25 GB/s per node
    bandwidth_bps = bandwidth_gbps * 1e9
    
    # All-reduce communication time
    comm_time = (total_bytes * 2) / bandwidth_bps  # Factor of 2 for bidirectional
    
    print(f"Model: {model_type}")
    print(f"Parameters: {params:,}")
    print(f"Communication volume: {total_bytes / 1e6:.1f} MB")
    print(f"All-reduce time: {comm_time * 1000:.2f} ms")
    
    # Rule of thumb: communication should be < 10% of compute time
    return comm_time

# Example outputs:
# VAE (50M params): ~100 MB, ~8 ms all-reduce
# Diffusion (200M params): ~400 MB, ~32 ms all-reduce
```

**Bandwidth Requirements and Network Topology:**

```
Single Node (8 GPUs):
├── GPU-GPU: NVLink/Infinity Fabric (600+ GB/s aggregate)
├── All-reduce pattern: Direct GPU-to-GPU communication
└── Optimal for: High-frequency parameter updates, small models

Multi-Node (80 GPUs across 10 nodes):
├── Intra-node: GPU-GPU via NVLink (600+ GB/s per node)
├── Inter-node: InfiniBand network (25 GB/s per node)  
├── All-reduce pattern: Hierarchical (intra-node + inter-node)
└── Optimal for: Large batch training, massive models
```

#### Fault Tolerance and Checkpointing

**Distributed Checkpointing for Multi-Node Training:**

```python
def distributed_checkpoint_save(accelerator, model, optimizer, epoch, 
                               checkpoint_dir):
    """
    Save checkpoint that can be resumed across different node configurations
    """
    if accelerator.is_main_process:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': accelerator.unwrap_model(model).state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accelerator_state': accelerator.state_dict(),
            'world_size': accelerator.num_processes,
            'config': model.config if hasattr(model, 'config') else None
        }
        
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

def distributed_checkpoint_load(accelerator, model, optimizer, checkpoint_path):
    """
    Load checkpoint with automatic device placement
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model state (unwrapped for compatibility)
    accelerator.unwrap_model(model).load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state 
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Resume from epoch
    start_epoch = checkpoint['epoch'] + 1
    
    print(f"Resumed from epoch {start_epoch}")
    return start_epoch
```

#### Integration in Python Code

The beauty of Accelerate is its simplicity while handling all the complex distributed training logic:

```python
# run_hydra_experiment.py
from accelerate import Accelerator
import torch.distributed as dist

@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(cfg: DictConfig) -> None:
    # Accelerate auto-detects SLURM environment
    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision="bf16"
    )
    
    print(f"Process {accelerator.process_index}/{accelerator.num_processes}")
    print(f"   Device: {accelerator.device}")
    print(f"   Node: {os.environ.get('SLURM_NODEID', 'unknown')}")
    
    # Test multi-node communication with all-reduce
    if torch.cuda.is_available() and dist.is_initialized():
        test_tensor = torch.ones(1, device=accelerator.device) * accelerator.process_index
        
        print(f"   Before all-reduce: {test_tensor.item():.0f} (rank {accelerator.process_index})")
        
        # All-reduce: sum values from all GPUs, then average
        dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
        
        if accelerator.is_main_process:
            expected_sum = sum(range(accelerator.num_processes))
            print(f"Communication test: {test_tensor.item():.0f} (expected: {expected_sum})")
            if abs(test_tensor.item() - expected_sum) < 1e-6:
                print("Multi-node all-reduce communication working!")
            else:
                print("Communication test failed!")
    
    # Your model training code here - all-reduce happens automatically
    model, optimizer = accelerator.prepare(model, optimizer)
    
    # During training, gradients are automatically averaged across all GPUs
    # No additional code required - Accelerate handles the complexity!
```

**Key Benefits of This Architecture:**

1. **Automatic Scaling**: Same code runs on 1 GPU or 1000+ GPUs
2. **Optimal Communication**: Ring all-reduce minimizes bandwidth usage
3. **Fault Tolerance**: Built-in checkpoint/resume capabilities  
4. **Memory Efficiency**: Mixed precision + gradient accumulation
5. **Load Balancing**: Distributed data sampling ensures even workload distribution

The framework abstracts away the complexity while providing optimal performance for X-ray diffraction model training at any scale.

## Part 2: Hydra Configuration Management for Scientific Experiments

### 2.1 Hierarchical Configuration Structure

Hydra enables clean separation of concerns through configuration composition:

```yaml
# configs/config.yaml
defaults:
  - model: vae_kl           # Choose your model architecture
  - experiment_type: train  # Training vs inference mode
  - data: full             # Dataset configuration
  - inference: default     # Inference parameters
  - _self_

hydra:
  run:
    dir: output/${hydra:runtime.choices.model}${experiment_type.test_suffix}/${now:%Y-%m-%d}/${now:%H-%M-%S}
```

### 2.2 Complete Model Configuration Reference

Understanding each model configuration is crucial for effective X-ray diffraction experiments. Here's a comprehensive breakdown of all available model configurations:

#### VAE with KL Divergence (Standard and High-Capacity)

**Standard VAE Configuration:**
```yaml
# configs/model/vae_kl.yaml
model_name: vae_kl
latent_channels: 4
latent_diff: false   # Set to true for latent diffusion
```

**High-Capacity VAE Configuration:**
```yaml
# configs/model/vae_kl_8.yaml  
model_name: vae_kl
latent_channels: 8   # Double the latent capacity
latent_diff: false
```

**When to use each:**
- `vae_kl` (4 channels): Standard experiments, faster training, lower memory usage
- `vae_kl_8` (8 channels): Complex diffraction patterns requiring higher representation capacity

#### Vector Quantized VAE (Standard and High-Capacity)

**Standard VQ-VAE Configuration:**
```yaml
# configs/model/vq.yaml  
model_name: vq
latent_channels: 4
latent_diff: false
```

**High-Capacity VQ-VAE Configuration:**
```yaml
# configs/model/vq_8.yaml
model_name: vq
latent_channels: 8   # Higher codebook capacity
latent_diff: false
```

**VQ-VAE Benefits for X-ray Diffraction:**
- Discrete latent representations ideal for crystallographic symmetries
- Better preservation of sharp diffraction peaks
- Reduced posterior collapse compared to standard VAEs

#### Direct Diffusion Model

**Standard Diffusion Configuration:**
```yaml
# configs/model/diff.yaml
model_name: diff
diff_epochs: 10
patch_size: 16          # Patch size for Vision Transformer backbone
vit_size: base          # [base, large, huge] - ViT model size
latent_diff: false      # Direct diffusion on image space
latent_channels: 3      # Input channels for diffraction data
depth: 8                # Transformer depth
dim_head: 1024          # Attention head dimension
heads: 8                # Number of attention heads
mlp_depth: 3            # MLP layers in transformer blocks

diffusion_kwargs:
  clamp_during_sampling: true  # Clamp values during generation
  num_sample_steps: 64         # Sampling steps for generation
  sigma_min: 0.002             # Minimum noise level
  sigma_max: 10                # Maximum noise level
  sigma_data: 0.5              # Data distribution std
  rho: 7                       # Sampling schedule parameter
  P_mean: -1.2                 # Training noise distribution mean
  P_std: 0.8                   # Training noise distribution std
  S_churn: 5                   # Stochastic sampling parameters
  S_tmin: 0.05                 # (see Karras et al. 2022)
  S_tmax: 50
  S_noise: 1.003
```

**Key Diffusion Parameters Explained:**
- `patch_size`: Determines how diffraction patterns are tokenized
- `vit_size`: Controls model capacity (base=86M, large=307M, huge=632M params)
- `num_sample_steps`: Higher values = better quality, slower generation
- `sigma_min/max`: Noise schedule range - critical for diffraction pattern quality

#### Latent Diffusion Model

**Latent Space Diffusion Configuration:**
```yaml
# configs/model/latent_diff.yaml
model_name: latent_diff
latent_diff: true        # Enable latent diffusion mode
diff_epochs: 10          # Diffusion-specific training epochs
patch_size: 16           # Patch size for latent space processing
vit_size: base           # ViT backbone size
latent_channels: 3       # Latent space channels
```

**Latent Diffusion Benefits:**
- Trains faster than direct diffusion (operates in compressed latent space)
- Better memory efficiency for high-resolution diffraction patterns
- Combines VAE's reconstruction quality with diffusion's generation diversity

### 2.3 Complete Dataset Configuration Reference

The framework supports multiple X-ray diffraction datasets with flexible preprocessing options:

#### Standard Dataset Configurations

**Full Dataset (522 samples):**
```yaml
# configs/data/full.yaml
data_id: 522             # Dataset identifier
avg_pooling: false       # No spatial pooling
topk: 1.0               # Use all diffraction peaks
data_path: /lustre/orion/mph121/proj-shared/datasets/peaknet20k
train_ratio: 0.8        # 80% training, 20% validation
seed: 42                # Reproducibility seed
batch_size: 2           # Default batch size
```

**Pooled Dataset (522 samples with averaging):**
```yaml
# configs/data/pooling.yaml
data_id: 522
avg_pooling: true        # Apply spatial averaging
topk: 1.0
data_path: /lustre/orion/mph121/proj-shared/datasets/peaknet20k
train_ratio: 0.8
seed: 42
batch_size: 2
```

**Alternative Dataset (422 samples):**
```yaml
# configs/data/full_422.yaml
data_id: 422             # Smaller dataset variant
avg_pooling: false
topk: 1.0
data_path: /lustre/orion/mph121/proj-shared/datasets/peaknet20k
train_ratio: 0.8
seed: 42
batch_size: 2
```

**Pooled Alternative Dataset:**
```yaml
# configs/data/pooling_422.yaml
data_id: 422
avg_pooling: true        # Combined smaller dataset + pooling
topk: 1.0
data_path: /lustre/orion/mph121/proj-shared/datasets/peaknet20k
train_ratio: 0.8
seed: 42
batch_size: 2
```

**Dataset Parameter Explanations:**
- `data_id`: 422 vs 522 refers to different experimental conditions or crystal systems
- `avg_pooling`: Reduces spatial resolution, useful for memory-constrained training
- `topk`: Fraction of strongest diffraction peaks to retain (1.0 = all peaks)
- `train_ratio`: Training/validation split ratio

### 2.4 Complete Training Configuration Reference

The framework provides specialized training configurations for different experimental scenarios:

#### Standard Training Configuration

**Base Training Setup:**
```yaml
# configs/experiment_type/train.yaml
batch_size: 2
test_pipeline: false     # Disable test mode
test_suffix: ""         # No suffix for standard training
num_epochs: 30
lr: 1e-4                # Learning rate
weight_decay: 1e-3      # L2 regularization
beta_recons: 0.5        # VAE reconstruction loss weight
recons_loss: iwmse      # [mse, l1, iwmse] reconstruction loss type
alpha_mse: 2.0          # MSE loss scaling factor
ema_decay: 0.9999       # Exponential moving average decay

# Training mode flags
train_vae_from_checkpoint: false
train_vae_from_scratch: true
train_diff_from_checkpoint: false
train_diff_from_scratch: true

# Model paths
pretrained_vae_path: null
pretrained_diff_path: null

# Annealing settings
use_annealing: true
annealing_shape: cosine  # [linear, cosine, logistic]
```

#### Loss-Specific Training Configurations

**MSE Loss Training:**
```yaml
# configs/experiment_type/train_mse.yaml
# Inherits from train.yaml but overrides:
recons_loss: mse        # Mean Squared Error loss
# Better for: Gaussian noise, smooth diffraction patterns
```

**L1 Loss Training:**
```yaml
# configs/experiment_type/train_l1.yaml
# Inherits from train.yaml but overrides:
recons_loss: l1         # L1 (Manhattan) loss
# Better for: Sparse diffraction patterns, outlier robustness
```

**IWMSE Loss Training:**
```yaml
# configs/experiment_type/train_iwmse.yaml
# Inherits from train.yaml but overrides:
recons_loss: iwmse      # Importance-weighted MSE
# Better for: Handling varying peak intensities in diffraction data
```

#### Test/Evaluation Configuration

**Test Pipeline Setup:**
```yaml
# configs/experiment_type/test.yaml
batch_size: 2
test_pipeline: true      # Enable test mode
test_suffix: "_test"     # Adds "_test" to output directory
num_epochs: 20          # Shorter for testing
lr: 1e-4
weight_decay: 1e-3
beta_recons: 0.5
recons_loss: iwmse
alpha_mse: 2.0
ema_decay: 0.9999

# Same training flags as base config
train_vae_from_checkpoint: false
train_vae_from_scratch: true
train_diff_from_checkpoint: false
train_diff_from_scratch: true

pretrained_vae_path: null
pretrained_diff_path: null

use_annealing: true
annealing_shape: cosine
```

**Key Training Parameter Explanations:**
- `beta_recons`: Controls VAE reconstruction vs. KL divergence trade-off
- `recons_loss`: Loss function choice affects peak preservation quality
- `alpha_mse`: Scaling factor for MSE component in composite losses
- `ema_decay`: Exponential moving average for stable training
- `annealing_shape`: KL annealing schedule shape affects training dynamics

### 2.5 Advanced Configuration Patterns

#### Configuration Composition Examples

**Multi-Model Comparison Study:**
```bash
# Compare all models with consistent settings
python run_hydra_experiment.py \
  model=vae_kl,vq,diff,latent_diff \
  experiment_type=train \
  data=full \
  training.num_epochs=20 \
  --multirun
```

**Loss Function Ablation Study:**
```bash
# Test different loss functions across models
python run_hydra_experiment.py \
  model=vae_kl \
  experiment_type=train_mse,train_l1,train_iwmse \
  data=full,pooling \
  --multirun
```

**Dataset Size Impact Study:**
```bash
# Compare performance on different dataset sizes
python run_hydra_experiment.py \
  model=vae_kl \
  data=full_422,full \
  training.batch_size=4,8,16 \
  --multirun
```

#### Runtime Configuration Overrides

**Memory-Constrained Training:**
```bash
# Reduce memory usage for limited GPU memory
python run_hydra_experiment.py \
  model=vae_kl \
  training.batch_size=1 \
  model.latent_channels=2 \
  data=pooling
```

**High-Capacity Training:**
```bash
# Maximum model capacity for best results
python run_hydra_experiment.py \
  model=vae_kl_8 \
  training.batch_size=8 \
  training.num_epochs=50 \
  data=full
```

**Checkpoint Resume Training:**
```bash
# Resume from checkpoint
python run_hydra_experiment.py \
  model=diff \
  training.train_diff_from_checkpoint=true \
  training.pretrained_diff_path=/path/to/checkpoint
```

#### Output Directory Structure

Hydra automatically organizes outputs based on configuration choices:

```
output/
├── vae_kl/                    # Model type
│   └── 2024-01-15/           # Date
│       └── 10-30-45/         # Time
│           ├── .hydra/       # Hydra metadata
│           ├── config.json   # Final model config
│           ├── checkpoints/  # Model checkpoints
│           └── samples/      # Generated samples
├── vae_kl_test/              # Test runs (with suffix)
└── diff/                     # Different model type
```

### 2.4 Data Configuration

```yaml
# configs/data/full.yaml
data_id: 522  # [422, 522] - Different X-ray diffraction datasets
avg_pooling: false
topk: 1.0
data_path: /lustre/orion/mph121/proj-shared/datasets/peaknet20k
train_ratio: 0.8
seed: 42
batch_size: 2
```

### 2.5 Using Hydra Overrides

The power of Hydra shines in command-line overrides:

```bash
# Train VAE with different loss functions
python run_hydra_experiment.py model=vae_kl experiment_type.recons_loss=mse

# Train diffusion model with larger batch size
python run_hydra_experiment.py model=diff training.batch_size=8

# Run latent diffusion with specific dataset
python run_hydra_experiment.py model=latent_diff data=pooling data.data_id=422

# Experiment sweep with different configurations
python run_hydra_experiment.py model=vq training.batch_size=4,8,16 -m
```

### 2.6 Configuration Best Practices

#### Model Selection Guide

**For X-ray Diffraction Tasks:**

| Model Type | Best Use Cases | Strengths | Considerations |
|------------|---------------|-----------|----------------|
| `vae_kl` | Pattern reconstruction, anomaly detection | Fast training, smooth latents | May blur sharp peaks |
| `vae_kl_8` | Complex patterns, high-quality reconstruction | Higher capacity | More memory, slower |
| `vq` | Discrete pattern analysis, classification | Sharp reconstructions, interpretable | Training instability |
| `vq_8` | High-resolution patterns | Best reconstruction quality | Highest memory usage |
| `diff` | High-quality synthesis, data augmentation | Best sample diversity | Slow training/inference |
| `latent_diff` | Efficient synthesis, style transfer | Fast diffusion training | Requires pre-trained VAE |

#### Loss Function Selection

**For Different Diffraction Characteristics:**

- **MSE (`train_mse`)**: Smooth patterns, Gaussian noise, overall pattern similarity
- **L1 (`train_l1`)**: Sparse peaks, outlier robustness, sharp feature preservation
- **IWMSE (`train_iwmse`)**: Varying peak intensities, weighted importance based on intensity

#### Dataset Configuration Strategy

**Memory vs. Quality Trade-offs:**

- Use `pooling` configs for memory-constrained environments
- Use `full` configs for maximum pattern resolution
- Choose `422` vs `522` based on your specific crystal systems
- Adjust `topk` parameter to focus on strongest diffraction peaks

#### Common Configuration Combinations

**Quick Start Combinations:**
```bash
# Development/Testing
python run_hydra_experiment.py model=vae_kl experiment_type=test data=pooling_422

# Production Training
python run_hydra_experiment.py model=vae_kl_8 experiment_type=train data=full

# High-Quality Generation
python run_hydra_experiment.py model=latent_diff experiment_type=train data=full training.num_epochs=50

# Memory-Efficient Training
python run_hydra_experiment.py model=vq experiment_type=train data=pooling training.batch_size=1
```

## Part 3: SLURM Integration for HPC Environments

### 3.1 ORNL Frontier Job Template

Here's a production-ready SLURM script for ORNL's Frontier supercomputer:

```bash
#!/bin/bash
# frontier.sbatch - ORNL Frontier Multi-Node Training

#SBATCH --output=slurm/hydra_experiment.%j.log
#SBATCH --error=slurm/hydra_experiment.%j.err
#SBATCH --account=AMPH121
#SBATCH --partition=batch
#SBATCH --qos=debug
#SBATCH --time=02:00:00
#SBATCH --job-name=hydra_vae_experiments
#SBATCH --nodes=10                    # 10 nodes
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=your-email@institution.edu
#SBATCH --ntasks-per-node=8           # 8 GPUs per node
#SBATCH --cpus-per-task=7             # 7 CPUs per task
#SBATCH --gpus-per-node=8             # 8 MI250X GPUs per node
#SBATCH --gpu-bind=closest

# Load Frontier modules
module load PrgEnv-amd
module load rocm/5.4.0
module load cray-mpich

# AMD GPU environment
export MPICH_GPU_SUPPORT_ENABLED=1
export FI_MR_CACHE_MONITOR=memhooks
export FI_CXI_RX_MATCH_MODE=software

# PyTorch/ROCm optimization
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:512
export HIP_FORCE_DEV_KERNARG=1
export NCCL_DEBUG=INFO
export TORCH_NCCL_BLOCKING_WAIT=0

# Multi-node networking
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
export MASTER_PORT=29500
export NODE_RANK=$SLURM_NODEID
export LOCAL_RANK=$SLURM_LOCALID
export WORLD_SIZE=80  # 10 nodes × 8 GPUs

# Fix networking issues
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_IB_DISABLE=1
export MIOPEN_USER_DB_PATH="/tmp/my-miopen-cache-$USER-$SLURM_JOB_ID"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

# OpenMP settings
export OMP_NUM_THREADS=7

# Experiment parameters
LATENT_CHANNELS=1
NUM_EPOCHS=20
BATCH_SIZE=32
LOSSES=("mse" "l1" "iwmse")
DATASETS=(422 522)

# Function to run distributed experiments
run_experiment() {
    local model=$1
    local dataset=$2
    local loss=$3
    local batch_size=$4
    local epochs=$5
    local channels=$6
    
    echo "Running: Model=$model, Dataset=$dataset, Loss=$loss"
    echo "Batch Size=$batch_size, Epochs=$epochs, Channels=$channels"
    
    # Use srun with accelerate for 80 GPUs total
    srun --ntasks=80 \
         --ntasks-per-node=8 \
         --cpus-per-task=7 \
         --gpus-per-node=8 \
         --gpu-bind=closest \
         accelerate launch \
         --config_file accelerate_config.yaml \
         --multi_gpu \
         --num_processes=80 \
         --num_machines=10 \
         --machine_rank=$SLURM_NODEID \
         --main_process_ip=$MASTER_ADDR \
         --main_process_port=29500 \
         run_hydra_experiment.py \
         model=$model \
         experiment_type=test \
         training.batch_size=$batch_size \
         training.num_epochs=$epochs \
         training.recons_loss=$loss \
         model.latent_channels=$channels \
         data.data_id=$dataset
}

# Run systematic experiments
for dataset in "${DATASETS[@]}"; do
    for loss in "${LOSSES[@]}"; do
        run_experiment vae_kl $dataset $loss $BATCH_SIZE $NUM_EPOCHS $LATENT_CHANNELS
        sleep 10  # System stability
    done
done

# Cleanup
rm -rf ${MIOPEN_USER_DB_PATH}
```

### 3.2 Simplified Multi-Node Script

For more straightforward multi-node execution without external accelerate launch:

```bash
#!/bin/bash
# run_diff_multi_node.sh - Simplified Multi-Node Diffusion

#SBATCH --account=mph121
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=8
#SBATCH --job-name=diff_522

EXPERIMENT_TYPE=${1:-train}

# Module loading and environment setup
module load PrgEnv-amd rocm/6.2.4 cray-mpich

# Environment variables
export MPICH_GPU_SUPPORT_ENABLED=1
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:512

# Network configuration with IPv4 enforcement
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_ADDR_IPv4=$(getent hosts "$MASTER_ADDR" | awk '{print $1}' | head -n 1)

export MASTER_ADDR="$MASTER_ADDR_IPv4"
export MASTER_PORT=$((23456 + ($SLURM_JOB_ID % 1000)))
export WORLD_SIZE=$SLURM_NTASKS
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_IB_DISABLE=1

# WandB configuration for offline logging
export WANDB_MODE=offline
export WANDB_DIR="$HOME/wandb_offline_logs"
mkdir -p $HOME/wandb_offline_logs

# Direct srun execution (Python script handles Accelerator internally)
srun --ntasks=$SLURM_NTASKS \
     --ntasks-per-node=$SLURM_NTASKS_PER_NODE \
     --cpus-per-task=$SLURM_CPUS_PER_TASK \
     bash -c "
export RANK=\$SLURM_PROCID
export LOCAL_RANK=\$SLURM_LOCALID
export WORLD_SIZE=\$SLURM_NTASKS
export MASTER_ADDR='$MASTER_ADDR'
export MASTER_PORT='$MASTER_PORT'
export NODE_RANK=\$SLURM_NODEID
export MACHINE_RANK=\$SLURM_NODEID

python run_hydra_experiment.py \
    model=diff \
    experiment_type=${EXPERIMENT_TYPE}_l1  \
    data=pooling
"
```

## Part 4: Complete Workflow Examples

### 4.1 Training a VAE for X-ray Diffraction Reconstruction

```bash
# Submit single-node VAE training
sbatch --nodes=1 --ntasks-per-node=8 --wrap="
accelerate launch --config_file accelerate_config/multigpu_config.yaml \
run_hydra_experiment.py \
model=vae_kl \
experiment_type=train \
data=full \
training.batch_size=16 \
training.num_epochs=50 \
training.recons_loss=iwmse
"
```

### 4.2 Training a Vector Quantized VAE

```bash
# VQ-VAE with different quantization levels
python run_hydra_experiment.py \
model=vq \
experiment_type=train \
training.batch_size=8 \
model.latent_channels=8 \
data.data_id=522
```

### 4.3 Training Direct Diffusion Models

```bash
# Multi-node diffusion model training
sbatch frontier.sbatch run_experiment diff 522 mse 16 30 3
```

### 4.4 Latent Diffusion Pipeline

```bash
# Two-stage latent diffusion: VAE + Diffusion
# Stage 1: Train VAE
python run_hydra_experiment.py \
model=vae_kl \
experiment_type=train \
training.num_epochs=30

# Stage 2: Train diffusion in latent space
python run_hydra_experiment.py \
model=latent_diff \
experiment_type=train \
training.pretrained_vae_path=/path/to/trained/vae
```

### 4.5 Hyperparameter Sweeps with Hydra Multirun

```bash
# Systematic loss function comparison
python run_hydra_experiment.py \
model=vae_kl \
experiment_type=train \
training.recons_loss=mse,l1,iwmse \
training.batch_size=4,8,16 \
--multirun
```

## Part 5: Monitoring and Debugging

### 5.1 WandB Integration

The training script automatically integrates with Weights & Biases:

```python
# Automatic logging setup
accelerator.init_trackers(
    args.model_name,
    config=cfg_dict,
    init_kwargs={
        "wandb": {
            "dir": out_dir, 
        }
    },
)
```

For offline environments (common in HPC):

```bash
export WANDB_MODE=offline
export WANDB_DIR="$HOME/wandb_offline_logs"

# Later sync when online
cd $HOME/wandb_offline_logs
wandb sync .
```

### 5.2 Debugging Multi-Node Issues

Common debugging steps:

```bash
# Check node connectivity
srun --ntasks=16 --ntasks-per-node=8 hostname

# Test GPU availability
srun --ntasks=16 --ntasks-per-node=8 python -c "
import torch
print(f'Node: {torch.cuda.device_count()} GPUs available')
"

# Verify network configuration
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "WORLD_SIZE: $WORLD_SIZE"
```

### 5.3 Performance Monitoring

```python
# Built-in communication test in training script
if dist.is_initialized():
    test_tensor = torch.ones(1, device=accelerator.device) * accelerator.process_index
    dist.all_reduce(test_tensor)
    
    if accelerator.is_main_process:
        expected_sum = sum(range(accelerator.num_processes))
        print(f"Communication test: {test_tensor.item():.0f} (expected: {expected_sum})")
        if abs(test_tensor.item() - expected_sum) < 1e-6:
            print("Multi-node communication working!")
```

## Part 6: Best Practices and Optimization

### 6.1 Memory Optimization

```yaml
# Optimized training configuration
training:
  batch_size: 4  # Start small, increase gradually
  gradient_accumulation_steps: 4  # Effective batch size = 4 × 4 = 16
  mixed_precision: bf16  # Reduces memory usage by ~50%
  
# PyTorch optimizations
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:512
```

### 6.2 Network Optimization for HPC

```bash
# Frontier-specific optimizations
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_IB_DISABLE=1
export GLOO_SOCKET_IFNAME=hsn0
export NCCL_SOCKET_IFNAME=hsn0
```

### 6.3 Efficient Data Loading

```python
# Optimized data pipeline
train_dataloader = DataLoader(
    train_dataset, 
    batch_size=args.batch_size, 
    shuffle=True,
    num_workers=4,  # Adjust based on CPU cores
    pin_memory=True,  # Faster GPU transfer
    persistent_workers=True  # Avoid worker respawning
)
```

### 6.4 Checkpointing Strategy

```python
# Distributed checkpointing
if accelerator.is_main_process:
    accelerator.save_state(f"checkpoint_epoch_{epoch}")
    
# Load for resuming
if resume_from_checkpoint:
    accelerator.load_state(checkpoint_path)
```

## Conclusion

This comprehensive guide demonstrates how to leverage Accelerate, Hydra, and SLURM for scaling X-ray diffraction modeling across multiple GPUs and compute nodes. The combination provides:

- **Seamless scaling**: From single GPU development to 800+ GPU production runs
- **Configuration management**: Clean separation of model, training, and data parameters
- **Reproducibility**: Systematic experiment tracking and configuration versioning
- **Production readiness**: Robust error handling and HPC integration

Whether you're training VAEs for diffraction pattern reconstruction, VQ-VAEs for discrete representation learning, or diffusion models for synthetic data generation, this framework provides the foundation for efficient, scalable scientific computing.

The modular design ensures your code remains maintainable while providing the flexibility to experiment with different architectures, loss functions, and training strategies—all essential for advancing the state-of-the-art in X-ray diffraction analysis.

## Citations

### Foundational Papers

**Variational Autoencoders:**
- Kingma, D. P., & Welling, M. (2013). *Auto-Encoding Variational Bayes*. arXiv preprint arXiv:1312.6114.
- Rezende, D. J., Mohamed, S., & Wierstra, D. (2014). *Stochastic Backpropagation and Approximate Inference in Deep Generative Models*. In International Conference on Machine Learning (pp. 1278-1286).

**β-VAE and Disentangled Representations:**
- Higgins, I., Matthey, L., Pal, A., Burgess, C., Glorot, X., Botvinick, M., ... & Lerchner, A. (2017). *β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework*. In International Conference on Learning Representations.

**Vector Quantized VAE:**
- van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017). *Neural Discrete Representation Learning*. In Advances in Neural Information Processing Systems (pp. 6306-6315).

**Diffusion Models:**
- Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models*. In Advances in Neural Information Processing Systems, 33, 6840-6851.
- Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., & Ganguli, S. (2015). *Deep Unsupervised Learning using Nonequilibrium Thermodynamics*. In International Conference on Machine Learning (pp. 2256-2265).
- Song, Y., & Ermon, S. (2019). *Generative Modeling by Estimating Gradients of the Data Distribution*. In Advances in Neural Information Processing Systems (pp. 11918-11930).

**Improved Diffusion Sampling:**
- Song, J., Meng, C., & Ermon, S. (2021). *Denoising Diffusion Implicit Models*. In International Conference on Learning Representations.
- Karras, T., Aittala, M., Aila, T., & Laine, S. (2022). *Elucidating the Design Space of Diffusion-Based Generative Models*. In Advances in Neural Information Processing Systems.

**Latent Diffusion Models:**
- Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). *High-Resolution Image Synthesis with Latent Diffusion Models*. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 10684-10695).

**Autoregressive Models:**
- Li, T., Tian, Y., Li, H., Deng, M., & He, K. (2024). *Autoregressive Image Generation without Vector Quantization*. In Advances in Neural Information Processing Systems (NeurIPS 2024). arXiv preprint arXiv:2406.11838.
- Li, Y., Liu, H., Wu, Q., Mu, F., Yang, J., Gao, J., ... & Lee, Y. J. (2024). *Autoregressive Image Generation using Residual Quantization*. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.
- Austin, J., Johnson, D. D., Ho, J., Tarlow, D., & van den Berg, R. (2021). *Structured Denoising Diffusion Models in Discrete State-Spaces*. In Advances in Neural Information Processing Systems.

**Noise Scheduling and Training Techniques:**
- Nichol, A., & Dhariwal, P. (2021). *Improved Denoising Diffusion Probabilistic Models*. In International Conference on Machine Learning (pp. 8162-8171).
- Chen, T., Zhang, R., & Hinton, G. (2023). *Analog Bits: Generating Discrete Data using Diffusion Models with Self-Conditioning*. In International Conference on Learning Representations.

### Technical Implementation Papers

**Distributed Training:**
- Rajbhandari, S., Rasley, J., Ruwase, O., & He, Y. (2020). *ZeRO: Memory optimizations toward training trillion parameter models*. In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (pp. 1-16).
- Li, M., Andersen, D. G., Park, J. W., Smola, A. J., Ahmed, A., Josifovski, V., ... & Su, B. Y. (2014). *Scaling Distributed Machine Learning with the Parameter Server*. In 11th USENIX Symposium on Operating Systems Design and Implementation (pp. 583-598).

**All-Reduce Communication:**
- Patarasuk, P., & Yuan, X. (2009). *Bandwidth optimal all-reduce algorithms for clusters of workstations*. Journal of Parallel and Distributed Computing, 69(2), 117-124.
- Thakur, R., Rabenseifner, R., & Gropp, W. (2005). *Optimization of collective communication operations in MPICH*. International Journal of High Performance Computing Applications, 19(1), 49-66.

**Mixed Precision Training:**
- Micikevicius, P., Narang, S., Alben, J., Diamos, G., Elsen, E., Garcia, D., ... & Woolley, C. (2018). *Mixed precision training*. In International Conference on Learning Representations.

### Domain-Specific Applications

**Scientific Computing with Deep Learning:**
- Baker, N., Alexander, F., Bremer, T., Hagberg, A., Kevrekidis, Y., Najm, H., ... & Zlatev, Z. (2019). *Workshop report on basic research needs for scientific machine learning: Core technologies for artificial intelligence*. USDOE Office of Science.

**X-ray Diffraction and Materials Science:**
- Park, W. B., Chung, J., Jung, J., Sohn, K., Singh, S. P., Pyo, M., ... & Sohn, K. S. (2017). *Classification of crystal structure using a convolutional neural network*. IUCrJ, 4(4), 486-494.
- Ziletti, A., Kumar, D., Scheffler, M., & Ghiringhelli, L. M. (2018). *Insightful classification of crystal structures using deep learning*. Nature Communications, 9(1), 2775.

**High-Performance Computing:**
- Dongarra, J., Beckman, P., Moore, T., Aerts, P., Aloisio, G., Andre, J. C., ... & Matsuoka, S. (2011). *The international exascale software project roadmap*. International Journal of High Performance Computing Applications, 25(1), 3-60.

### Framework and Library References

**Accelerate:**
- Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., ... & Rush, A. M. (2020). *Transformers: State-of-the-art natural language processing*. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations (pp. 38-45).

**Hydra Configuration Management:**
- Yadan, O. (2019). *Hydra - A framework for elegantly configuring complex applications*. GitHub repository. https://github.com/facebookresearch/hydra

**SLURM Workload Manager:**
- Yoo, A. B., Jette, M. A., & Grondona, M. (2003). *SLURM: Simple Linux Utility for Resource Management*. In Workshop on Job Scheduling Strategies for Parallel Processing (pp. 44-60). Springer.

## Additional Resources

- [Accelerate Documentation](https://huggingface.co/docs/accelerate)
- [Hydra Configuration Framework](https://hydra.cc/)
- [SLURM Workload Manager](https://slurm.schedmd.com/)
- [ORNL Frontier User Guide](https://docs.olcf.ornl.gov/systems/frontier_user_guide.html)
- [PyTorch Distributed Training](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)