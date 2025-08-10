import copy
import contextlib
import torch
import accelerate

def _generic_unwrap(model):
    """
    Works for torch DDP, FSDP, DeepSpeed, and many wrappers that expose `.module`.
    Falls back to the given model if no wrapping is present.
    """
    # Try Accelerate utility if available (no hard dependency).
    try:
        from accelerate.utils import extract_model_from_parallel
        return extract_model_from_parallel(model)
    except Exception:
        pass

    m = model
    # Unroll nested .module wrappers if present.
    while hasattr(m, "module"):
        m = m.module
    return m

class EMA:
    def __init__(self, model, decay=0.999, device=None, dtype=None, accelerator=None, unwrap_fn=None):
        """
        Args:
            model: possibly-wrapped model (Accelerate/DDP/FSDP/DeepSpeed/etc.)
            decay: EMA decay (e.g., 0.999–0.9999)
            device: where to keep EMA weights (default: 'cpu')
            dtype: cast EMA weights to this dtype (default: keep param dtype; commonly torch.float32)
            accelerator: optional accelerate.Accelerator to use its unwrap_model
            unwrap_fn: optional callable(model)->base_module to unwrap custom wrappers
        """
        self.decay = float(decay)
        self._unwrap = (
            (unwrap_fn if unwrap_fn is not None else
             (accelerator.unwrap_model if accelerator is not None else _generic_unwrap))
        )

        base = self._unwrap(model)
        keep_dev = device if device is not None else 'cpu'

        # Track only trainable params
        self._param_names = [n for n, p in base.named_parameters() if p.requires_grad]

        # Shadow params
        self.shadow = {}
        for n, p in base.named_parameters():
            if p.requires_grad:
                t = p.detach().to(keep_dev)
                if dtype is not None:
                    t = t.to(dtype)
                self.shadow[n] = t.clone()

        # Copy buffers (e.g., BN running stats) verbatim; no decay.
        self.buffers = {}
        for n, b in base.named_buffers():
            tb = b.detach().to(keep_dev).clone()
            if dtype is not None and tb.is_floating_point():
                tb = tb.to(dtype)
            self.buffers[n] = tb

    @torch.no_grad()
    def update(self, model):
        """Update EMA from a (possibly wrapped) live model."""
        base = self._unwrap(model)
        d = self.decay

        for n, p in base.named_parameters():
            if n in self.shadow:
                s = self.shadow[n]
                # Move source to shadow device to avoid device mismatch
                src = p.detach().to(s.device)
                if s.dtype.is_floating_point and src.dtype != s.dtype:
                    src = src.to(s.dtype)
                s.mul_(d).add_(src, alpha=(1.0 - d))

        # Keep buffers in sync (copy, no decay)
        for n, b in base.named_buffers():
            if n in self.buffers:
                dst = self.buffers[n]
                src = b.detach().to(dst.device)
                if dst.dtype.is_floating_point and src.dtype != dst.dtype:
                    src = src.to(dst.dtype)
                dst.copy_(src)

    @contextlib.contextmanager
    def apply_to(self, model):
        """
        Temporarily swap EMA weights/buffers into the given (possibly wrapped) model.
        Useful for evaluation/saving without permanently overwriting training weights.
        """
        base = self._unwrap(model)

        # Stash originals
        orig = {}

        with torch.no_grad():
            for n, p in base.named_parameters():
                if n in self.shadow:
                    orig[n] = p.data.clone()
                    p.data.copy_(self.shadow[n].to(p.device, dtype=p.dtype))
            for n, b in base.named_buffers():
                if n in self.buffers:
                    if n not in orig:
                        orig[n] = b.data.clone()
                    b.data.copy_(self.buffers[n].to(b.device, dtype=b.dtype))

        try:
            yield
        finally:
            with torch.no_grad():
                for n, p in base.named_parameters():
                    if n in orig:
                        p.data.copy_(orig[n])
                for n, b in base.named_buffers():
                    if n in orig:
                        b.data.copy_(orig[n])