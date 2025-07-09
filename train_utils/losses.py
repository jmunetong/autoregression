
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