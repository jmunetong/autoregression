
import torch

class IntensityWeightedMSELoss(nn.Module):

    def __init__(self, alpha=2.0):
        super(IntensityWeightedMSELoss, self).__init__()
        self.alpha = alpha

    def forward(self, input, target):
        weights = 1 + self.alpha * target
        return ((input - target)**2 * weights).mean()