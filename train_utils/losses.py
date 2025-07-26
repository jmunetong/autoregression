
from torch import nn
from torch.nn import functional as F
import torch
    
class IntensityWeightedMSELoss(nn.Module):
    def __init__(self, alpha=2.0, kernel_size=3, eps=1e-6):
        super(IntensityWeightedMSELoss, self).__init__()
        self.alpha = alpha
        self.kernel_size = kernel_size
        self.eps = eps

    def compute_local_contrast(self, x):
        # x: shape (B, 1, H, W)
        mean = F.avg_pool2d(x, self.kernel_size, stride=1, padding=self.kernel_size // 2)
        contrast = torch.abs(x - mean)
        return contrast

    def forward(self, input, target):
        if input.dim() == 3:
            input = input.unsqueeze(1)
            target = target.unsqueeze(1)
        contrast = self.compute_local_contrast(target)
        weights = 1.0 + self.alpha * contrast
        return ((input - target) ** 2 * weights).mean()
    



import torch
import torch.nn.functional as F

def adaptive_weighted_loss(pred, target, kernel_size=15, weight_factor=2.0):
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
