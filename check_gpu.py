#!/usr/bin/env python3
"""
GPU Health Check Script for PyTorch
Tests all available GPUs for proper functionality including:
- Basic CUDA availability
- Memory allocation and operations
- Multi-GPU communication
- Performance benchmarking
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import time
import gc
import sys
import os
from typing import List, Dict, Any

def print_separator(title: str):
    """Print a formatted separator with title"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def basic_gpu_info():
    """Check basic GPU information and CUDA availability"""
    print_separator("BASIC GPU INFORMATION")
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. Exiting...")
        return False
    
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    # List all available GPUs
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {props.name}")
        print(f"  Memory: {props.total_memory / 1e9:.1f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Multiprocessors: {props.multi_processor_count}")
    
    return True

def test_individual_gpu(device_id: int) -> Dict[str, Any]:
    """Test individual GPU functionality"""
    results = {
        'device_id': device_id,
        'basic_ops': False,
        'memory_ops': False,
        'neural_net': False,
        'memory_gb': 0.0,
        'performance_tflops': 0.0,
        'errors': []
    }
    
    try:
        device = torch.device(f'cuda:{device_id}')
        torch.cuda.set_device(device)
        
        # Test 1: Basic tensor operations
        print(f"\n🔧 Testing GPU {device_id} - Basic Operations")
        try:
            a = torch.randn(1000, 1000, device=device)
            b = torch.randn(1000, 1000, device=device)
            c = torch.matmul(a, b)
            torch.cuda.synchronize()
            results['basic_ops'] = True
            print(f"  ✅ Basic tensor operations: PASSED")
        except Exception as e:
            results['errors'].append(f"Basic ops failed: {str(e)}")
            print(f"  ❌ Basic tensor operations: FAILED - {e}")
        
        # Test 2: Memory operations
        print(f"🔧 Testing GPU {device_id} - Memory Operations")
        try:
            # Test different memory sizes
            sizes = [1000, 5000, 10000]
            max_size = 0
            
            for size in sizes:
                try:
                    test_tensor = torch.randn(size, size, device=device)
                    memory_used = torch.cuda.memory_allocated(device) / 1e9
                    max_size = size
                    del test_tensor
                    torch.cuda.empty_cache()
                except RuntimeError:
                    break
            
            results['memory_ops'] = True
            results['memory_gb'] = torch.cuda.get_device_properties(device_id).total_memory / 1e9
            print(f"  ✅ Memory operations: PASSED (max tensor: {max_size}x{max_size})")
            print(f"  📊 Total GPU memory: {results['memory_gb']:.1f} GB")
            
        except Exception as e:
            results['errors'].append(f"Memory ops failed: {str(e)}")
            print(f"  ❌ Memory operations: FAILED - {e}")
        
        # Test 3: Neural network operations
        print(f"🔧 Testing GPU {device_id} - Neural Network Operations")
        try:
            model = nn.Sequential(
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 10)
            ).to(device)
            
            # Forward pass
            x = torch.randn(64, 512, device=device)
            output = model(x)
            
            # Backward pass
            loss = torch.sum(output)
            loss.backward()
            
            torch.cuda.synchronize()
            results['neural_net'] = True
            print(f"  ✅ Neural network operations: PASSED")
            
        except Exception as e:
            results['errors'].append(f"Neural net failed: {str(e)}")
            print(f"  ❌ Neural network operations: FAILED - {e}")
        
        # Test 4: Performance benchmark
        print(f"🔧 Testing GPU {device_id} - Performance Benchmark")
        try:
            # Matrix multiplication benchmark
            size = 2048
            num_ops = 10
            
            torch.cuda.synchronize()
            start_time = time.time()
            
            for _ in range(num_ops):
                a = torch.randn(size, size, device=device)
                b = torch.randn(size, size, device=device)
                c = torch.matmul(a, b)
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            # Calculate TFLOPS (approximate)
            total_ops = num_ops * 2 * size**3  # 2 * N^3 for matrix multiplication
            elapsed_time = end_time - start_time
            tflops = (total_ops / elapsed_time) / 1e12
            
            results['performance_tflops'] = tflops
            print(f"  📊 Performance: {tflops:.2f} TFLOPS")
            
        except Exception as e:
            results['errors'].append(f"Performance test failed: {str(e)}")
            print(f"  ❌ Performance test: FAILED - {e}")
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        gc.collect()
        
    except Exception as e:
        results['errors'].append(f"Device setup failed: {str(e)}")
        print(f"❌ GPU {device_id} setup failed: {e}")
    
    return results

def test_multi_gpu_communication():
    """Test multi-GPU communication and data transfer"""
    print_separator("MULTI-GPU COMMUNICATION TEST")
    
    if torch.cuda.device_count() < 2:
        print("⚠️  Only one GPU available, skipping multi-GPU tests")
        return True
    
    try:
        # Test peer-to-peer communication
        print("🔧 Testing GPU-to-GPU communication...")
        
        device_0 = torch.device('cuda:0')
        device_1 = torch.device('cuda:1')
        
        # Create tensors on different devices
        tensor_0 = torch.randn(1000, 1000, device=device_0)
        tensor_1 = torch.randn(1000, 1000, device=device_1)
        
        # Test data transfer
        tensor_0_to_1 = tensor_0.to(device_1)
        tensor_1_to_0 = tensor_1.to(device_0)
        
        # Test computation across devices
        result = torch.matmul(tensor_0_to_1, tensor_1)
        torch.cuda.synchronize()
        
        print("  ✅ GPU-to-GPU communication: PASSED")
        
        # Test DataParallel if available
        if torch.cuda.device_count() >= 2:
            print("🔧 Testing DataParallel...")
            model = nn.Linear(100, 50)
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.cuda()
            
            x = torch.randn(32, 100).cuda()
            output = model(x)
            print("  ✅ DataParallel: PASSED")
        
        return True
        
    except Exception as e:
        print(f"❌ Multi-GPU communication failed: {e}")
        return False

def test_distributed_training():
    """Test basic distributed training setup"""
    print_separator("DISTRIBUTED TRAINING TEST")
    
    if torch.cuda.device_count() < 2:
        print("⚠️  Need at least 2 GPUs for distributed training test")
        return True
    
    try:
        # This is a basic check - full distributed testing requires multiple processes
        print("🔧 Checking distributed training compatibility...")
        
        # Check if NCCL backend is available
        if dist.is_nccl_available():
            print("  ✅ NCCL backend: AVAILABLE")
        else:
            print("  ⚠️  NCCL backend: NOT AVAILABLE")
        
        # Check environment variables commonly used in distributed training
        env_vars = ['WORLD_SIZE', 'RANK', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT']
        for var in env_vars:
            value = os.environ.get(var, 'Not set')
            print(f"  📋 {var}: {value}")
        
        print("  ℹ️  For full distributed testing, run with torchrun or accelerate launch")
        
        return True
        
    except Exception as e:
        print(f"❌ Distributed training check failed: {e}")
        return False

def run_stress_test(duration_seconds: int = 30):
    """Run a stress test on all GPUs"""
    print_separator(f"STRESS TEST ({duration_seconds} seconds)")
    
    if torch.cuda.device_count() == 0:
        print("❌ No GPUs available for stress test")
        return False
    
    print(f"🔧 Running stress test on all {torch.cuda.device_count()} GPUs...")
    
    processes = []
    start_time = time.time()
    
    try:
        # Run computation on each GPU
        tensors = []
        for i in range(torch.cuda.device_count()):
            device = torch.device(f'cuda:{i}')
            tensor = torch.randn(2000, 2000, device=device)
            tensors.append(tensor)
        
        # Continuous computation
        operations = 0
        while time.time() - start_time < duration_seconds:
            for i, tensor in enumerate(tensors):
                with torch.cuda.device(i):
                    result = torch.matmul(tensor, tensor)
                    operations += 1
            
            if operations % 100 == 0:
                elapsed = time.time() - start_time
                print(f"  ⏱️  {elapsed:.1f}s - {operations} operations completed")
        
        # Synchronize all devices
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device=i)
        
        print(f"  ✅ Stress test completed: {operations} operations in {duration_seconds}s")
        return True
        
    except Exception as e:
        print(f"❌ Stress test failed: {e}")
        return False

def generate_report(gpu_results: List[Dict[str, Any]]):
    """Generate a summary report"""
    print_separator("SUMMARY REPORT")
    
    total_gpus = len(gpu_results)
    working_gpus = sum(1 for r in gpu_results if r['basic_ops'] and r['memory_ops'] and r['neural_net'])
    
    print(f"📊 Total GPUs: {total_gpus}")
    print(f"✅ Working GPUs: {working_gpus}")
    print(f"❌ Failed GPUs: {total_gpus - working_gpus}")
    
    if working_gpus > 0:
        avg_performance = sum(r['performance_tflops'] for r in gpu_results if r['performance_tflops'] > 0) / working_gpus
        total_memory = sum(r['memory_gb'] for r in gpu_results)
        
        print(f"📈 Average Performance: {avg_performance:.2f} TFLOPS")
        print(f"💾 Total GPU Memory: {total_memory:.1f} GB")
    
    print(f"\n📋 Detailed Results:")
    for result in gpu_results:
        status = "✅ WORKING" if result['basic_ops'] and result['memory_ops'] and result['neural_net'] else "❌ FAILED"
        print(f"  GPU {result['device_id']}: {status}")
        
        if result['errors']:
            for error in result['errors']:
                print(f"    ⚠️  {error}")
    
    # Overall assessment
    if working_gpus == total_gpus:
        print(f"\n🎉 ALL GPUS ARE WORKING PROPERLY!")
    elif working_gpus > 0:
        print(f"\n⚠️  {total_gpus - working_gpus} GPU(s) have issues. Check individual results above.")
    else:
        print(f"\n💥 ALL GPUS FAILED! Check CUDA installation and drivers.")

def main():
    """Main function to run all GPU tests"""
    print("🚀 Starting GPU Health Check...")
    
    # Basic GPU info
    if not basic_gpu_info():
        sys.exit(1)
    
    # Test each GPU individually
    gpu_results = []
    for i in range(torch.cuda.device_count()):
        print_separator(f"TESTING GPU {i}")
        result = test_individual_gpu(i)
        gpu_results.append(result)
    
    # Multi-GPU tests
    test_multi_gpu_communication()
    test_distributed_training()
    
    # Optional stress test
    run_stress = input("\n🤔 Run stress test? (y/N): ").lower().strip() == 'y'
    if run_stress:
        duration = input("Enter duration in seconds (default 30): ").strip()
        duration = int(duration) if duration.isdigit() else 30
        run_stress_test(duration)
    
    # Generate final report
    generate_report(gpu_results)

if __name__ == "__main__":
    main()