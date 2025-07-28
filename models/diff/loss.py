import torch
import torch.nn.functional as F

from einops import rearrange

def adaptive_weighted_loss(pred, target, kernel_size=15, weight_factor=2.0, reshape_to_img=None, reshape_to_seq=None, batch_size=None):
    """
    Compute adaptive weighted L1 loss based on local contrast of the target image.

    Args:
        pred (Tensor): Predicted tensor (flattened or image-shaped)
        target (Tensor): Target tensor (same shape as pred)
        kernel_size (int): Neighborhood size for local std computation
        weight_factor (float): Contrast scaling factor
        reshape_to_img (Callable, optional): Function to reshape (B*N, D) -> (B, C, H, W)
        reshape_to_seq (Callable, optional): Function to reshape back to (B*N, D)
    """
    # Reshape if needed
    assert reshape_to_img is not None
    assert reshape_to_seq is not None
    assert batch_size is not None, "Batch size must be provided for reshaping"
    assert pred.shape == target.shape, "Predicted and target tensors must have the same shape"

    if reshape_to_img is not None:
        target  = rearrange(target, '(b n) d -> b n d', b=batch_size)
        pred = rearrange(pred, '(b n) d -> b n d', b=batch_size)
        target = reshape_to_img(target)
        pred = reshape_to_img(pred)
    else:
        target = target
        pred = pred

    # Compute local contrast from target
    padding = kernel_size // 2

    pad_top = pad_left = kernel_size // 2
    pad_bottom = kernel_size - pad_top - 1
    pad_right = kernel_size - pad_left - 1

    padded = F.pad(
        target,
        (pad_left, pad_right, pad_top, pad_bottom),  # (left, right, top, bottom)
        mode='reflect'
    )
    local_mean = F.avg_pool2d(padded, kernel_size, stride=1)

    local_var = F.avg_pool2d(
        F.pad((target - local_mean) ** 2, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect'),
        kernel_size,
        stride=1
    )
    local_std = torch.sqrt(local_var + 1e-6)

    # Normalize contrast to get weights
    weight_map = 1.0 + weight_factor * (local_std / local_std.mean())

    # Compute L1 loss
    base_loss = torch.abs(pred - target)

    # Apply weights
    weighted_loss = base_loss * weight_map

    # Optional: flatten again if needed
    if reshape_to_seq is not None:
        weighted_loss = reshape_to_seq(weighted_loss)

    return weighted_loss.mean()