import glob
import os
import logging

import yaml
import torch
from tqdm import tqdm

import torch
from transformers import get_cosine_schedule_with_warmup
from torch.optim import AdamW

from train_utils.annealing import Annealer
from train_utils.configs import MODELS, RECONS_LOSS, init_configure_model, init_configure_diffusion
from train_utils.ema import EMA

TEST_LEGNTH = 1

def cleanup_old_checkpoints(path, keep_last=3):
    """Keep only the last N training checkpoints to save disk space"""
    checkpoints = sorted(glob.glob(f"{path}/raw_epoch_*"))
    for old_checkpoint in checkpoints[:-keep_last]:
        os.remove(old_checkpoint)

def get_last_checkpoint(path):
    """
    Get the last checkpoint from the specified path.
    Returns the path to the last checkpoint or None if no checkpoints are found.
    """
    checkpoints = sorted(glob.glob(f"{path}/raw_epoch_*"))
    if not checkpoints:
        return None
    return checkpoints[-1]


class BaseTrainer():
    def __init__(self, model, args,  accelerator, len_dataloader=None):
        self.accelerator = accelerator
        self.current_epoch = 0
        self.model = model
        self.ema_model = EMA(model=model, decay=getattr(args, "ema_decay", 0.9999), device=accelerator.device, dtype=torch.float32, accelerator=accelerator)
        self.optimizer = self._init_optimizer()
        assert len_dataloader is not None, "len_train_train_dataloader must be provided"

        # Scheduler parameters
        num_training_steps = len_dataloader * args.num_epochs
        num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup
        # preparing configurations for the model
        self.scheduler = self._init_scheduler(num_training_steps, num_warmup_steps)

        # Prepare the model, optimizer, and scheduler with the accelerator
        self.model, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.scheduler)
        self.model_vae = self.model #TODO: FIX THIS VALUE

        # Initialize losses
        self.recons_loss = RECONS_LOSS[args.recons_loss]
        self.use_annealing = args.use_annealing
        if self.use_annealing:
            total_steps = args.num_epochs
            shape = args.annealing_shape
            baseline = 0.0
            cyclical = False
            disable = False

            self.annealer = Annealer(total_steps, shape, baseline, cyclical, disable)

    def get_model(self, with_accelerator=True):
        return self.model if with_accelerator else self.accelerator.unwrap_model(self.model)

    def run_trainer(self, train_dataloader, experiment_dict, directory):
        """
        Run the training loop for the model.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def get_model_config(self):
        return self.model_config
 
    def _init_optimizer(self):
        optimizer = AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        return optimizer
    
        # Scheduler
    def _init_scheduler(self, num_training_steps, num_warmup_steps):
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps)
        return scheduler
         
    def load_weights(self, directory,):

        # 1. Load model weights (after .prepare, so we can unwrap)
        directory_raw = get_last_checkpoint(directory)
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.from_pretrained(directory_raw)
        checkpoint_path = os.path.join(directory_raw, 'checkpoint.pt')

        self.load_ema_weights(directory)
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.accelerator.device) #TODO: SHOULD THIS ACTUALLY BE ON GPU OR CPU

            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])

            epoch = checkpoint.get('epoch', 0)
            loss = checkpoint.get('loss', None)
            return epoch, loss
        else:
            raise FileNotFoundError(f"checkpoint.pt not found in {directory}")
        
       
    def load_ema_weights(self, directory):
        """
        Load the EMA weights from the specified directory.
        """
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory {directory} does not exist.")
        
        ema_path = os.path.join(directory, "ema_model.pt")
        if not os.path.exists(ema_path):
            raise FileNotFoundError(f"EMA model file {ema_path} does not exist.")
        
        ema_state_dict = torch.load(ema_path, map_location=self.accelerator.device)
        self.ema_model.load_state_dict(ema_state_dict)
        
    def save_raw_checkpoint(self, root_dir: str, *, epoch: int, step: int, train_loss: float) -> str:
        """
        Save the most recent RAW training snapshot:
        - HF weights (consolidated via Accelerate)
        - optimizer & scheduler state
        - small metadata file
        Returns the directory path written to.
        """
        out_dir = os.path.join(root_dir, f"raw_epoch_{epoch}")
        os.makedirs(out_dir, exist_ok=True)

        # sync all ranks before saving
        self.accelerator.wait_for_everyone()

        # consolidated state dict (FSDP/ZeRO/DDP-safe)
        state_dict = self.accelerator.get_state_dict(self.model)

        if self.accelerator.is_main_process:
            base = self.accelerator.unwrap_model(self.model)
            base.save_pretrained(
                out_dir,
                is_main_process=True,
                save_function=self.accelerator.save,
                state_dict=state_dict,
            )
            torch.save(
                {
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "epoch": int(epoch),
                    "step": int(step),
                    "train_loss": float(train_loss),
                },
                os.path.join(out_dir, "checkpoint.pt"),
            )
            with open(os.path.join(root_dir, "LATEST_RAW.txt"), "w") as f:
                f.write(out_dir)

        cleanup_old_checkpoints(root_dir, keep_last=3)
        self.accelerator.wait_for_everyone()
        return out_dir
    

    def save_ema_check_point(self, root_dir):
        state_dict = self.ema_model.state_dict()
        if not state_dict:
            if self.accelerator.is_main_process:
                print("[save_ema_checkpoint] Skipped: EMA model is empty.")
            return None
        torch.save(state_dict, os.path.join(root_dir, "ema_model.pt"))
        logging.info(f"EMA model saved to {root_dir}/ema_model.pt")
    
     
    def _save_experiment_config(self, experiment_dict, directory):
        with open(os.path.join(directory, "experiment_config.yml"), "w") as f:
            yaml.dump(experiment_dict, f, default_flow_style=False)


class TrainerVQ(BaseTrainer):
    def __init__(self, args, accelerator, len_dataloader=None):
        self.args = args
        self.model_config = init_configure_model(args)
        super().__init__(model=self._init_model(), args=args, accelerator=accelerator, len_dataloader=len_dataloader)

    def compute_loss(self, out, batch, beta_recons):
        """
        Compute the loss for the outputs.
        This method should be overridden by subclasses to implement specific loss calculations.
        """
        loss_i  = out.commit_loss
        recons = out.sample
        recon_loss_i = self.recons_loss(recons, batch)
        loss_i = beta_recons * recon_loss_i + loss_i
        return loss_i, recon_loss_i
    
    def is_wrapped(self):
        """Check if model is wrapped by distributed training frameworks"""
        from torch.nn.parallel import DistributedDataParallel
        from torch.distributed.fsdp import FullyShardedDataParallel
        
        # Check for common wrapper types
        wrapper_types = (DistributedDataParallel,)
        
        # Add FSDP if available
        try:
            wrapper_types += (FullyShardedDataParallel,)
        except ImportError:
            pass
        
        return isinstance(self.model, wrapper_types) or hasattr(self.model, 'module')

    def step(self, batch, i, epoch, experiment_dict, directory):
        # ALL processes run the same forward pass
        out = self.model(batch, return_dict=True)
        
        # Only main process handles logging/saving (non-blocking operations)
        if i == 0 and epoch == 0 and self.accelerator.is_main_process:
            try:
                # Get model for encoding (separate from main forward pass)
                if self.is_wrapped():
                    unwrapped_model = self.accelerator.unwrap_model(self.model)
                    with torch.no_grad():  # Don't interfere with gradients
                        latents = unwrapped_model.encode(batch[:1], return_dict=True).latents
                else:
                    with torch.no_grad():
                        latents = self.model.encode(batch, return_dict=True).latents

                # Save configuration
                experiment_dict["input_shape"] = list(batch.shape[1:])
                experiment_dict["latent_shape"] = list(latents.shape[1:])
                self._save_experiment_config(experiment_dict, directory)
                print(f"Batch shape: {batch.shape}")
                print(f"Latent sample shape: {latents.shape}")
                
            except Exception as e:
                print(f"Warning: Configuration saving failed: {e}")
        
        return out

    def run_train(self, train_dataloader, val_dataloader, experiment_dict, directory):
        best_loss = float('inf')
        beta_recons = self.args.beta_recons
        
        for epoch in range(self.args.num_epochs if not self.args.test_pipeline else TEST_LEGNTH):
            if self.accelerator.is_main_process:
                print(f"Epoch {epoch+1}/{self.args.num_epochs}")    
            
            epoch_loss = 0.0
            epoch_recon_loss = 0.0
            self.model.train()

            for i, batch in tqdm(enumerate(train_dataloader), 
                    total=len(train_dataloader), 
                    desc="Training",
                    disable=not self.accelerator.is_main_process):
                
                
                self.optimizer.zero_grad()
                batch = batch.contiguous()
                
                if self.args.test_pipeline and i > TEST_LEGNTH:
                    break
                    
                if i == 0 and epoch == 0 and self.accelerator.is_main_process:
                    print(f"Batch shape: {batch.shape}")

                # Forward pass
                out = self.step(batch, i, epoch, experiment_dict, directory)
                
                # Loss computation
                loss_i, recon_loss_i = self.compute_loss(out, batch, beta_recons)

                # Backward pass
                self.accelerator.wait_for_everyone()
                self.accelerator.backward(loss_i)
                self.optimizer.step()
                self.scheduler.step()

                # Add process ID to debug EMA
                self.ema_model.update(self.model)
                                
                # Track metrics (local accumulation)
                epoch_loss += loss_i.item()
                epoch_recon_loss += recon_loss_i.item()
                self.accelerator.wait_for_everyone()
            # Update epoch metrics with batch averages
    
            epoch_loss /= len(train_dataloader)
            epoch_recon_loss /= len(train_dataloader)
            self.current_epoch += 1
            all_epoch_losses = self.accelerator.gather(torch.tensor(epoch_loss, device=self.accelerator.device))
            global_avg_loss = all_epoch_losses.mean().item()

            # Only main process prints and saves
            if self.accelerator.is_main_process:
                print(f"Global average epoch {epoch + 1} train loss: {global_avg_loss}")
                self.save_raw_checkpoint(directory, epoch=epoch+1, step=i+1, train_loss=global_avg_loss)
            
            # Validation (all processes participate)
            val_loss = self.validate_with_ema(val_dataloader)
            
            if self.accelerator.is_main_process:
                print(f"Validation Loss: {val_loss:.4f}")
            
            # Save best model
            is_best = val_loss < best_loss
            if is_best:
                best_loss = val_loss
                if self.accelerator.is_main_process:
                    print(f"New best loss: {best_loss}")
                    self.save_ema_check_point(directory)
            
            # Optional: Sync all processes at the end of epoch
            self.accelerator.wait_for_everyone()
    
    def validate_with_ema(self, val_loader):
        self.model.eval()

        # Apply EMA weights temporarily
        with self.ema_model.apply_to(self.model):
            total_loss = 0.0
            num_batches = 0
            
            with torch.no_grad():
                for _, batch in tqdm(enumerate(val_loader), 
                    total=len(val_loader), 
                    desc="Validation EMA",
                    disable=not self.accelerator.is_main_process):
                    outputs = self.model(batch, return_dict=True)
                    loss, _ = self.compute_loss(outputs, batch, self.args.beta_recons)
                    
                    # Just accumulate local losses
                    total_loss += loss.item()
                    num_batches += 1
                
        
            # Calculate local average
            if num_batches > 0:
                avg_loss = total_loss / num_batches
            else:
                avg_loss = 0.0
            
            # Gather the averaged losses from all processes and compute global average
            avg_loss_tensor = torch.tensor(avg_loss, device=self.accelerator.device)
                  # CRITICAL: Wait for all processes to finish the validation loop
            self.accelerator.wait_for_everyone()
            gathered_losses = self.accelerator.gather(avg_loss_tensor)
            global_avg_loss = gathered_losses.mean().item()
            # Final sync to ensure all processes are done
            self.accelerator.wait_for_everyone()
        
        return global_avg_loss
    
    def _init_model(self):
        assert self.args.model_name.startswith("vq"), "Model name must start with 'vq' for VQTrainer"
        model = MODELS[self.args.model_name](**self.model_config)
        return model

class TrainerVAE(BaseTrainer):
    def __init__(self, args, accelerator, len_dataloader=None):
        self.args = args
        self.model_config = init_configure_model(args)
        super().__init__(model=self._init_model(), 
                         args=args, 
                         accelerator=accelerator, 
                         len_dataloader=len_dataloader)
        
    
        
    def step(self, batch, i, epoch, experiment_dict, directory):
        """
        Perform a single step of the VAE training.
        """
        ## Encoding Step
        if not hasattr(self.model, 'module'):
            # Model is wrapped (DDP, FSDP, etc.) - unwrap it
            posterior = self.model.encode(batch).latent_dist
        else:
            posterior = self.accelerator.unwrap_model(self.model).encode(batch).latent_dist
            # Model is not wrapped - use directly
            
        mu_posterior = posterior.mean
        logvar_posterior = posterior.logvar
        
        # Decoding step
        posterior_sample = posterior.sample()
        if i == 0 and epoch == 0 and self.accelerator.is_main_process:
            experiment_dict["input_shape"] = list(batch.shape[1:])
            experiment_dict["latent_shape"] = list(posterior_sample.shape[1:])
            self._save_experiment_config(experiment_dict, directory)

        self.accelerator.wait_for_everyone()
        if not hasattr(self.model, 'module'):
            recon_i = self.model.decode(posterior_sample)
        else:
            recon_i = self.accelerator.unwrap_model(self.model).decode(posterior_sample)

        return recon_i, logvar_posterior, mu_posterior
    
    def compute_loss(self, recon_i, logvar_posterior, mu_posterior, batch, batch_size, beta_recons):
        # Loss Function Computation
        kl_loss_i = -0.5 * torch.sum(1 + logvar_posterior - mu_posterior.pow(2) - torch.exp(logvar_posterior))
        kl_loss_i /= batch_size
        kl_loss_i = self.annealer(kl_loss_i) if self.use_annealing else kl_loss_i
         
        recon_loss_i = self.recons_loss(recon_i.sample, batch)
        loss_i = beta_recons * recon_loss_i + kl_loss_i 
        return loss_i, recon_loss_i, kl_loss_i
    

    def run_train(self, train_dataloader, val_dataloader, experiment_dict, directory):
        self.model.train()
        best_loss = float('inf')
        beta_recons = self.args.beta_recons

        for epoch in range(self.args.num_epochs if not self.args.test_pipeline else TEST_LEGNTH):
            if self.accelerator.is_main_process:
                print(f"Epoch {epoch+1}/{self.args.num_epochs}")     
            epoch_loss = 0.0
            epoch_kl_loss = 0.0
            epoch_recon_loss = 0.0
            self.model.train()

            for i, batch in tqdm(enumerate(train_dataloader), 
                     total=len(train_dataloader), 
                     desc="Training",
                     disable=not self.accelerator.is_main_process):
                # Important before starting one forward pass
                if self.args.test_pipeline and i > TEST_LEGNTH:
                    break
                if i == 0 and epoch == 0 and self.accelerator.is_main_process:
                    print(f"Batch shape: {batch.shape}")
                self.optimizer.zero_grad()
                batch = batch.contiguous()
                # ic("Running forward step")
                recon_i, logvar_posterior, mu_posterior = self.step(batch, i, epoch, experiment_dict, directory)
                # ic("computing loss")
                loss_i, recon_loss_i, kl_loss_i = self.compute_loss(recon_i, logvar_posterior, mu_posterior, batch, batch_size=batch.size(0), beta_recons=beta_recons)
        
                self.accelerator.backward(loss_i)
                
                # Step optimizer after accumulating gradients
                self.optimizer.step()
                self.scheduler.step()
                self.ema_model.update(self.model)
                del recon_i, mu_posterior, logvar_posterior

                # Track metrics
                epoch_loss += loss_i.item()
                epoch_kl_loss += kl_loss_i.item()
                epoch_recon_loss += recon_loss_i.item()

            self.accelerator.wait_for_everyone()  
            # Update epoch metrics with batch averages
            epoch_loss /= len(train_dataloader)
            epoch_kl_loss /= len(train_dataloader)
            epoch_recon_loss /= len(train_dataloader)
            self.current_epoch += 1
            
                    
            all_epoch_losses = self.accelerator.gather(torch.tensor(epoch_loss, device=self.accelerator.device))
            global_avg_loss = all_epoch_losses.mean().item()

            # Only main process prints
            if self.accelerator.is_main_process:
                print(f"Global average epoch {epoch + 1} train loss: {global_avg_loss}")
            self.accelerator.wait_for_everyone()
            self.accelerator.log({"epoch": epoch+1, "loss": epoch_loss, "recon_loss": epoch_recon_loss, "kl_loss": epoch_kl_loss})
            val_loss = self.validate_with_ema(val_dataloader)

            if self.accelerator.is_main_process:
                print(f"Validation Loss: {val_loss:.4f}")
            # Saving Best model
            if val_loss < best_loss:
                best_loss = val_loss
                if self.accelerator.is_main_process:
                    print(f"New best loss: {best_loss}")
                self.save_ema_check_point(directory)
            self.accelerator.wait_for_everyone()


    def validate_with_ema(self, val_loader):
        self.model.eval()

        # Apply EMA weights temporarily
        with self.ema_model.apply_to(self.model):
            total_loss = 0.0
            num_batches = 0
            
            with torch.no_grad():
                if self.accelerator.is_main_process:
                    print("Validating with EMA weights...")
                for batch in tqdm(val_loader, desc="Validation", total=len(val_loader)):

                    recon_i, logvar_posterior, mu_posterior = self.step(batch, i=2, epoch=2, experiment_dict=None, directory=None)

                    loss = self.compute_loss(recon_i=recon_i, logvar_posterior=logvar_posterior, mu_posterior=mu_posterior, batch=batch, batch_size=batch.size(0), beta_recons=self.args.beta_recons)[0]
                    
                    # Just accumulate local losses (no gather needed here)
                    total_loss += loss.item()
                    num_batches += 1
        
            # Calculate local average
            if num_batches > 0:
                avg_loss = total_loss / num_batches
            else:
                avg_loss = 0.0
            
            # Gather the averaged losses from all processes and compute global average
            avg_loss_tensor = torch.tensor(avg_loss, device=self.accelerator.device)
            gathered_losses = self.accelerator.gather(avg_loss_tensor)
            global_avg_loss = gathered_losses.mean().item()
                    
        return global_avg_loss
        

    def _init_model(self):
        assert self.args.model_name.startswith("vae"), "Model name must start with 'vae' for VAETrainer"
        model = MODELS[self.args.model_name](**self.model_config)
        return model

class TrainerDiffusionNonVAE(BaseTrainer):
    def __init__(self, args, accelerator, len_train_dataloader=None, input_shape=None):
        self.args = args
        assert input_shape is not None, "input_shape must be provided"
        self.image_shape = input_shape
        diff_model, self.model_config = init_configure_diffusion(
            vit_size=args.vit_size,
            patch_size=args.patch_size,
            input_shape=input_shape[-1] # Assuming input shape is (1, height, width)
        )
        super().__init__(model=diff_model, args=args, accelerator=accelerator, len_dataloader=len_train_dataloader)
        
        self.accelerator.wait_for_everyone()
        
    @staticmethod
    def unwrap(model):
        return model.module if hasattr(model, "module") else model
    
    def run_train(self, train_dataloader, val_dataloader, experiment_dict, directory):

        best_loss = float('inf')
        for epoch in range(self.args.diff_epochs if not self.args.test_pipeline else TEST_LEGNTH):
            if self.accelerator.is_main_process:
                print(f"Epoch {epoch+1}/{self.args.diff_epochs}")     
            epoch_loss = 0.0
            self.model.train()
            for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Training"):          
                self.optimizer.zero_grad()
                assert batch.shape[-1] == self.image_shape[-1], f"Batch shape {batch.shape} does not match expected shape {self.image_shape}"
                if self.args.test_pipeline and i > TEST_LEGNTH:
                    break
                loss_i = self.model(batch)
                if i == 0 and epoch == 0 and self.accelerator.is_main_process:
                    self._save_experiment_config(experiment_dict, directory)
                
                self.accelerator.backward(loss_i)

                self.optimizer.step()
                self.scheduler.step()
                self.ema_model.update(self.model)
                self.optimizer.zero_grad()
                
                self.accelerator.wait_for_everyone()

                epoch_loss += loss_i.item()
            epoch_loss /= len(train_dataloader)

            self.accelerator.wait_for_everyone()  
            # Update epoch metrics with batch averages
                        
            all_epoch_losses = self.accelerator.gather(torch.tensor(epoch_loss, device=self.accelerator.device))
            global_avg_loss = all_epoch_losses.mean().item()

            if self.accelerator.is_main_process:
                print(f"Epoch {epoch+1}, Loss: {global_avg_loss}")
            self.accelerator.log({"epoch": epoch+1, "loss": global_avg_loss})
            self.current_epoch +=1
            val_loss = self.validate_with_ema(val_dataloader)
            self.save_raw_checkpoint(directory, epoch=epoch+1, step=i+1, train_loss=global_avg_loss)

            if self.accelerator.is_main_process:
                print(f"Validation Loss: {val_loss:.4f}")

            # Saving Best model
            if val_loss < best_loss:
                best_loss = val_loss
                if self.accelerator.is_main_process:
                    print(f"New best loss: {best_loss}")
                self.save_ema_check_point(directory)

            self.accelerator.wait_for_everyone()
    
        print('training complete')

    
    def save_raw_checkpoint(self, root_dir: str, *, epoch: int, step: int, train_loss: float) -> str:
        """
        Save the most recent RAW training snapshot:
        - HF weights (consolidated via Accelerate)
        - optimizer & scheduler state
        - small metadata file
        Returns the directory path written to.
        """
        out_dir = os.path.join(root_dir, f"raw_epoch_{epoch}")
        os.makedirs(out_dir, exist_ok=True)

        # sync all ranks before saving
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            base = self.accelerator.unwrap_model(self.model)
            torch.save(
                {"model": base.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "epoch": int(epoch),
                    "step": int(step),
                    "train_loss": float(train_loss),
                },
                os.path.join(out_dir, "checkpoint.pt"),
            )
            with open(os.path.join(root_dir, "LATEST_RAW.txt"), "w") as f:
                f.write(out_dir)

        cleanup_old_checkpoints(root_dir, keep_last=3)
        self.accelerator.wait_for_everyone()
        return out_dir

    def get_diff_model(self):
        """
        Returns the diffusion model.
        """
        return self.unwrap(self.model)
    
    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def step(self, batch):
        return self.model(batch)
    

    def validate_with_ema(self, val_loader):
        self.model.eval()
        
        # Apply EMA weights temporarily
        with self.ema_model.apply_to(self.model):
            total_loss = 0
            total_samples = 0
            
            with torch.no_grad():
                for i, batch in tqdm(enumerate(val_loader), desc="Validation", total=len(val_loader), disable=not self.accelerator.is_main_process):
                    if self.args.test_pipeline and i > TEST_LEGNTH:
                        break
                    
                    loss = self.step(batch)
                    batch_size = self.get_batch_size(batch)  # You'll need to implement this
                    
                    # Accumulate loss weighted by batch size
                    total_loss += loss.item() * batch_size
                    total_samples += batch_size
            
            # Create tensors for reduction across processes
            total_loss_tensor = torch.tensor(total_loss, device=self.accelerator.device)
            total_samples_tensor = torch.tensor(total_samples, device=self.accelerator.device)
            
            # Reduce across all processes
            total_loss_all = self.accelerator.reduce(total_loss_tensor, reduction="sum")
            total_samples_all = self.accelerator.reduce(total_samples_tensor, reduction="sum")
            
            # Compute global average
            avg_loss = total_loss_all.item() / total_samples_all.item()
            
            # Only print on main process
            if self.accelerator.is_main_process:
                print(f"Validation Loss: {avg_loss:.4f}")
        
        return avg_loss


    def get_batch_size(self, batch):
        """Helper function to get batch size from your batch format"""
        # Adjust this based on your batch structure
        if isinstance(batch, dict):
            # If batch is a dictionary, get size from first tensor
            first_key = next(iter(batch))
            return batch[first_key].shape[0]
        elif isinstance(batch, (list, tuple)):
            # If batch is a list/tuple, get size from first element
            return batch[0].shape[0]
        else:
            # If batch is a tensor directly
            return batch.shape[0]
    
    def load_weights(self, directory,):

        # 1. Load model weights (after .prepare, so we can unwrap)
        directory_raw = get_last_checkpoint(directory)
        self.model = self.accelerator.unwrap_model(self.model)
        checkpoint_path = os.path.join(directory_raw, 'checkpoint.pt')

        self.load_ema_weights(directory)
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.accelerator.device) #TODO: SHOULD THIS ACTUALLY BE ON GPU OR CPU
            self.model.load_state_dict(checkpoint['model'])
            self.model = self.accelerator.prepare(self.model)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            epoch = checkpoint.get('epoch', 0)
            loss = checkpoint.get('loss', None)
            self.accelerator.wait_for_everyone()
            return epoch, loss
        else:
            raise FileNotFoundError(f"checkpoint.pt not found in {directory}")

class TrainerDiffusion(TrainerDiffusionNonVAE):
    def __init__(self, args, model_vae,  accelerator, len_train_dataloader=None, input_shape=None):

        self.model_vae = self.accelerator.prepare(model_vae)  
        self.image_shape = input_shape
        self.model_vae.eval()
    
        if accelerator.is_main_process:
            encoding_shape = self._get_prediction_shape_image()
        else:
            encoding_shape = None
        self.encoding_shape = accelerator.broadcast_object(encoding_shape, src=0)
        del encoding_shape
        super().__init__(args, accelerator, len_train_train_dataloader=len_train_dataloader, input_shape=self.encoding_shape[-1])
              
        self.accelerator.wait_for_everyone()

    def _get_prediction_shape_image(self):
        sample = torch.randn(self.image_shape).to(self.accelerator.device)
        self.model_vae.eval()
        with torch.no_grad():
            out = self.model_vae.encode(sample).latent_dist.sample()
        return out.shape[1:]

    @staticmethod
    def unwrap(model):
        return model.module if hasattr(model, "module") else model
    
    def step(self, batch):

        latents = self.model_vae.encode(batch).latent_dist.sample()
        loss_i = self.model(latents)
        return loss_i


    def run_train(self, train_dataloader, val_dataloader, experiment_dict, directory):
        best_loss = float('inf')
        # self.model_vae.eval()
        self.model_vae.train()
        self.model.train()

        for epoch in range(self.args.diff_epochs if not self.args.test_pipeline else TEST_LEGNTH):
            if self.accelerator.is_main_process:
                print(f"Epoch {epoch+1}/{self.args.diff_epochs}")     
            epoch_loss = 0.0
            self.model.train()
            for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Training"):          
                
                loss_i = self.step(batch)
                if i == 0 and epoch == 0 and self.accelerator.is_main_process:
                    self._save_experiment_config(experiment_dict, directory)
                   
                self.accelerator.backward(loss_i)

                self.optimizer.step()
                self.optimizer.zero_grad()

                if self.is_main:
                    self.unwrap(self.ema_model).update()
                # self.accelerator.wait_for_everyone()
                # self.unwrap(self.ema_model).update()
                self.accelerator.wait_for_everyone()
                
                epoch_loss += loss_i.item()
            
            
            if self.accelerator.is_main_process:
                print(f"Epoch {epoch+1}, Loss: {epoch_loss}")
            self.accelerator.log({"epoch": epoch+1, "loss": epoch_loss})
            val_loss = self.validate_with_ema(val_dataloader)
            # Saving Best model
            if val_loss < val_loss:
                best_loss = epoch_loss
                if self.accelerator.is_main_process:
                    print(f"New best loss: {best_loss}")
                self.save_ema_check_point(directory)

            self.accelerator.wait_for_everyone()
    
        print('training complete')

    
    def get_vae_model(self):
        """
        Returns the VAE model.
        """
        return self.unwrap(self.model_vae)
    
    


        
