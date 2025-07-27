import torch
import torch.nn.functional as F

def adaptive_weighted_loss(pred, target, kernel_size=16, weight_factor=2.0):
    """
    Compute adaptive weighted loss based on local contrast.
    
    Args:
        pred: Predicted tensor
        target: Target tensor
        kernel_size: Size of the kernel for local contrast computation
        weight_factor: Factor to scale the weight map
    
    Returns:
        Weighted loss value
    """
    # Compute local mean using average pooling
    padding = kernel_size // 2
    local_mean = F.avg_pool2d(
        F.pad(target, (padding, padding, padding, padding), mode='reflect'),
        kernel_size,
        stride=1
    )
    
    # Compute local standard deviation
    local_var = F.avg_pool2d(
        F.pad((target - local_mean)**2, (padding, padding, padding, padding), mode='reflect'),
        kernel_size,
        stride=1
    )
    local_std = torch.sqrt(local_var + 1e-6)
    
    # Normalize to create weight map
    weight_map = 1.0 + weight_factor * (local_std / local_std.mean())
    
    # Compute base L1 loss
    base_loss = torch.abs(pred - target)
    
    # Apply weights to loss
    weighted_loss = base_loss * weight_map
    
    return weighted_loss.mean()