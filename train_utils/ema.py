import copy
import contextlib
import torch
import logging

def _generic_unwrap(model):
    """
    Works for torch DDP, FSDP, DeepSpeed, and many wrappers that expose `.module`.
    Falls back to the given model if no wrapping is present.
    """

    try:
        from accelerate.utils import extract_model_from_parallel
        return extract_model_from_parallel(model)
    except (ImportError, AttributeError) as e:
        logging.debug(f"Accelerate extraction failed: {e}")
    except Exception as e:
        logging.warning(f"Unexpected error in accelerate extraction: {e}")

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
        self.accelerator = accelerator  # Store accelerator reference
        self._unwrap = (
            unwrap_fn if unwrap_fn is not None else
            (accelerator.unwrap_model if accelerator is not None else _generic_unwrap)
        )

        base = self._unwrap(model)
        keep_dev = device if device is not None else 'cpu'

        # Track only trainable params
        self._param_names = []
        # Shadow params
        self.shadow = {}
        for n, p in base.named_parameters():
            if p.requires_grad:
                t = p.detach().to(keep_dev)
                if dtype is not None:
                    t = t.to(dtype)
                self.shadow[n] = t.clone()
                self._param_names.append(n)

        # Copy buffers (e.g., BN running stats) verbatim; no decay.
        self.buffers = {}
        for n, b in base.named_buffers():
            tb = b.detach().to(keep_dev).clone()
            if dtype is not None and tb.is_floating_point():
                tb = tb.to(dtype)
            self.buffers[n] = tb

        # Fix: Check if accelerator exists and is main process
        if accelerator is not None and accelerator.is_main_process:
            print(f"EMA initialized with {len(self.shadow)} parameters and {len(self.buffers)} buffers")
        elif accelerator is None:
            print(f"EMA initialized with {len(self.shadow)} parameters and {len(self.buffers)} buffers")

    @torch.no_grad()
    def update(self, model):
        """Update EMA from a (possibly wrapped) live model."""
        try:
            base = self._unwrap(model)
            d = self.decay

            # Update parameters
            for n, p in base.named_parameters():
                if p.requires_grad and n in self.shadow:  # Add requires_grad check
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
                    
        except Exception as e:
            # Log the error but don't crash training
            if self.accelerator is not None and self.accelerator.is_main_process:
                print(f"Error in EMA update: {e}")
            raise e

    @contextlib.contextmanager
    def apply_to(self, model):
        """
        Temporarily swap EMA weights/buffers into the given (possibly wrapped) model.
        Useful for evaluation/saving without permanently overwriting training weights.
        
        IMPORTANT: In distributed training, all processes must enter/exit this context
        manager together to maintain synchronization.
        """
        # Ensure all processes start applying EMA weights together
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
            
        base = self._unwrap(model)
        # Stash originals
        orig_params = {}
        orig_buffers = {}

        with torch.no_grad():
            # Store and replace parameters
            for n, p in base.named_parameters():
                if n in self.shadow:
                    orig_params[n] = p.data  # Store reference, not clone
                    p.data = self.shadow[n].to(p.device, dtype=p.dtype)
            
            # Store and replace buffers
            for n, b in base.named_buffers():
                if n in self.buffers:
                    orig_buffers[n] = b.data  # Store reference, not clone
                    b.data = self.buffers[n].to(b.device, dtype=b.dtype)

        # Ensure all processes have applied EMA weights before proceeding
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()

        try:
            yield
        finally:
            with torch.no_grad():
                # Restore parameters
                for n, p in base.named_parameters():
                    if n in orig_params:
                        p.data = orig_params[n]
                
                # Restore buffers
                for n, b in base.named_buffers():
                    if n in orig_buffers:
                        b.data = orig_buffers[n]
            
            # Ensure all processes have restored original weights together
            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()

    def state_dict(self):
        """Return state dict for saving/loading EMA state."""
        return {
            'decay': self.decay,
            'shadow': self.shadow,
            'buffers': self.buffers,
            'param_names': self._param_names
        }

    def load_state_dict(self, state_dict):
        """Load EMA state from state dict."""
        self.decay = state_dict['decay']
        self.shadow = state_dict['shadow']
        self.buffers = state_dict['buffers']
        self._param_names = state_dict['param_names']

    def copy_to(self, model):
        """
        Permanently copy EMA weights to model (useful for final model saving).
        Unlike apply_to, this doesn't restore original weights.
        """
        base = self._unwrap(model)
        
        with torch.no_grad():
            for n, p in base.named_parameters():
                if n in self.shadow:
                    p.data.copy_(self.shadow[n].to(p.device, dtype=p.dtype))
            
            for n, b in base.named_buffers():
                if n in self.buffers:
                    b.data.copy_(self.buffers[n].to(b.device, dtype=b.dtype))