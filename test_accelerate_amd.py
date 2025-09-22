#!/usr/bin/env python3
"""
Example of using Accelerate with AMD GPUs on ORNL Frontier
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from accelerate import Accelerator, DistributedDataParallelKwargs
import time
import os

def check_amd_gpu_setup():
    """Check if AMD GPU setup is working"""
    print("🔍 Checking AMD GPU Setup for Accelerate")
    print(f"PyTorch version: {torch.__version__}")
    print(f"ROCm/CUDA available: {torch.cuda.is_available()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Check environment variables
    env_vars = ['ROCR_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'SLURM_GPUS_PER_NODE']
    for var in env_vars:
        print(f"{var}: {os.environ.get(var, 'Not set')}")

def create_dummy_model_and_data():
    """Create a simple model and dataset for testing"""
    # Simple neural network
    model = nn.Sequential(
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 10),
        nn.LogSoftmax(dim=1)
    )
    
    # Dummy dataset
    X = torch.randn(1000, 784)
    y = torch.randint(0, 10, (1000,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    return model, dataloader

def test_basic_accelerate():
    """Test basic Accelerate functionality with AMD GPUs"""
    print("\n🚀 Testing Basic Accelerate with AMD GPUs")
    
    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision="fp16",  # Test mixed precision with AMD
        gradient_accumulation_steps=2
    )
    
    print(f"Device: {accelerator.device}")
    print(f"Process index: {accelerator.process_index}")
    print(f"Local process index: {accelerator.local_process_index}")
    print(f"Num processes: {accelerator.num_processes}")
    print(f"Is main process: {accelerator.is_main_process}")
    
    # Create model and data
    model, dataloader = create_dummy_model_and_data()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Prepare everything with accelerator
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    
    # Test training loop
    model.train()
    total_loss = 0
    start_time = time.time()
    
    for batch_idx, (data, target) in enumerate(dataloader):
        with accelerator.accumulate(model):
            output = model(data)
            loss = nn.functional.nll_loss(output, target)
            
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
        
        if batch_idx >= 10:  # Just test a few batches
            break
    
    end_time = time.time()
    
    # Gather results across all processes
    total_loss = accelerator.gather_for_metrics(torch.tensor(total_loss)).mean()
    
    if accelerator.is_main_process:
        print(f"✅ Training completed successfully!")
        print(f"Average loss: {total_loss:.4f}")
        print(f"Time taken: {end_time - start_time:.2f} seconds")
    
    return True

def test_distributed_accelerate():
    """Test distributed training with multiple AMD GPUs"""
    print("\n🚀 Testing Distributed Accelerate with Multiple AMD GPUs")
    
    # Configure for distributed training
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    
    accelerator = Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_steps=2,
        kwargs_handlers=[ddp_kwargs]
    )
    
    if accelerator.is_main_process:
        print(f"Running distributed training on {accelerator.num_processes} AMD GPUs")
    
    # Create larger model for distributed testing
    model = nn.Sequential(
        nn.Linear(784, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    
    # Create larger dataset
    X = torch.randn(10000, 784)
    y = torch.randint(0, 10, (10000,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    
    # Prepare with accelerator
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    
    # Training loop
    model.train()
    epoch_loss = 0
    num_batches = 0
    
    start_time = time.time()
    
    for batch_idx, (data, target) in enumerate(dataloader):
        with accelerator.accumulate(model):
            output = model(data)
            loss = nn.functional.cross_entropy(output, target)
            
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        if batch_idx >= 50:  # Test reasonable number of batches
            break
        
        # Progress reporting from main process only
        if accelerator.is_main_process and batch_idx % 10 == 0:
            print(f"Batch {batch_idx}: Loss = {loss.item():.4f}")
    
    end_time = time.time()
    
    # Gather final metrics
    final_loss = epoch_loss / num_batches
    final_loss = accelerator.gather_for_metrics(torch.tensor(final_loss)).mean()
    
    if accelerator.is_main_process:
        print(f"✅ Distributed training completed!")
        print(f"Final average loss: {final_loss:.4f}")
        print(f"Training time: {end_time - start_time:.2f} seconds")
        print(f"Batches per second: {num_batches / (end_time - start_time):.2f}")
    
    return True

def test_accelerate_features():
    """Test specific Accelerate features with AMD GPUs"""
    print("\n🧪 Testing Specific Accelerate Features")
    
    accelerator = Accelerator(mixed_precision="fp16")
    
    # Test 1: Gradient clipping
    print("Testing gradient clipping...")
    model, dataloader = create_dummy_model_and_data()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    
    data, target = next(iter(dataloader))
    output = model(data)
    loss = nn.functional.nll_loss(output, target)
    accelerator.backward(loss)
    
    # Gradient clipping
    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    print("✅ Gradient clipping works")
    
    # Test 2: Save/Load state
    print("Testing save/load state...")
    if accelerator.is_main_process:
        accelerator.save_state("test_checkpoint")
    
    accelerator.wait_for_everyone()
    # Note: Loading would require careful coordination in distributed setting
    print("✅ Save state works")
    
    # Test 3: Gathering tensors
    print("Testing tensor gathering...")
    test_tensor = torch.randn(4, device=accelerator.device)
    gathered = accelerator.gather(test_tensor)
    if accelerator.is_main_process:
        print(f"Gathered tensor shape: {gathered.shape}")
        print("✅ Tensor gathering works")
    
    return True

def main():
    """Main function to test Accelerate with AMD GPUs"""
    print("🔥 Testing Accelerate with AMD GPUs on Frontier")
    print("=" * 60)
    
    # Check setup
    check_amd_gpu_setup()
    
    if not torch.cuda.is_available():
        print("❌ No GPUs available. Make sure you're on a compute node with GPU allocation.")
        print("💡 Try: srun --gpus=8 python test_accelerate_amd.py")
        return False
    
    try:
        # Test basic functionality
        test_basic_accelerate()
        
        # Test distributed training if multiple GPUs
        if torch.cuda.device_count() > 1:
            test_distributed_accelerate()
        else:
            print("⚠️ Only 1 GPU available, skipping distributed tests")
        
        # Test specific features
        test_accelerate_features()
        
        print("\n🎉 All Accelerate tests passed with AMD GPUs!")
        return True
        
    except Exception as e:
        print(f"\n❌ Accelerate test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()