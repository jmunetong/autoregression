import os
from pyexpat import model
import copy 
import shutil
import contextlib

import yaml
import torch
from tqdm import tqdm

from icecream import ic

import torch
from transformers import get_cosine_schedule_with_warmup
from torch.optim import AdamW

from train_utils.annealing import Annealer
from train_utils.configs import MODELS, RECONS_LOSS, init_configure_model, init_configure_diffusion
from train_utils.ema import EMA

TEST_LEGNTH = 1


class BaseTrainer():
    def __init__(self, model, args,  accelerator, len_dataloader=None):
        self.accelerator = accelerator
        self.current_epoch = 0
        self.model = model
        self.ema_model = EMA(model=model, decay=getattr("ema_decay", 0.9999), device=accelerator.device, dtype=torch.float32, accelerator=accelerator)
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

    # def load_weights(self, directory):

    #     ##TODO: REMOVE THIS FUNCTION. IT MIGHT NOT BE NECESSARY
    #     """
    #     Load the model weights from the specified directory.
    #     """
    #     if not os.path.exists(directory):
    #         raise FileNotFoundError(f"Directory {directory} does not exist.")
        
    #     checkpoint_path = os.path.join(directory, "checkpoint.pt")
    #     if not os.path.exists(checkpoint_path):
    #         raise FileNotFoundError(f"Checkpoint file {checkpoint_path} does not exist.")
        
    #     checkpoint = torch.load(checkpoint_path, map_location=self.accelerator.device)

    #     self.model.load_state_dict(checkpoint['model'])
    #     if 'ema_model' in checkpoint:
    #         self.ema_model.load_state_dict(checkpoint['ema_model'])
    #     if 'optimizer' in checkpoint:
    #         self.optimizer.load_state_dict(checkpoint['optimizer'])
         
    def load_weights(self, directory):

        # 1. Load model weights (after .prepare, so we can unwrap)
        directory_raw = os.path.join(directory, "raw_last")
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.from_pretrained(directory_raw)

        # 2. Load optimizer/scheduler/epoch/loss from checkpoint.pt
        #TODO: MODIFY THE DIRECTORY TO REFER TO THE RAW MODEL
        checkpoint_path = os.path.join(directory, 'checkpoint.pt')
        directory_ema = os.path.join(directory, "ema_best")
        self.load_ema_weights(directory_ema)
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
        #TODO: Modify to load the EMA weights directory'
        directory = os.path.join(directory, "ema_best")
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory {directory} does not exist.")
        
        ema_path = os.path.join(directory, "ema_model.pt")
        if not os.path.exists(ema_path):
            raise FileNotFoundError(f"EMA model file {ema_path} does not exist.")
        
        ema_state_dict = torch.load(ema_path, map_location=self.accelerator.device)
        self.ema_model.load_state_dict(ema_state_dict)
        
    def save_vae(self, directory):
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(
        directory,
        is_main_process=self.accelerator.is_main_process,
        save_function=self.accelerator.save)

    def save_raw_checkpoint(self, root_dir: str, *, epoch: int, step: int, train_loss: float) -> str:
        """
        Save the most recent RAW training snapshot:
        - HF weights (consolidated via Accelerate)
        - optimizer & scheduler state
        - small metadata file
        Returns the directory path written to.
        """
        out_dir = os.path.join(root_dir, "raw_last")
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

        # optional barrier so non-main ranks don't touch the folder early
        self.accelerator.wait_for_everyone()
        return out_dir
    
    def save_ema_checkpoint(
        self,
        root_dir: str,
        *,
        epoch: int,
        step: int,
        ema_val: float | None = None,
        keep_history: bool = False,
        tag: str | None = None,
    ) -> str | None:
        """
        Save an EMA snapshot (weights only) for inference/deploy.
        - If keep_history=False (default): writes/overwrites <root>/ema_best/
        - If keep_history=True: writes <root>/ema_best_step{step}/ (keeps all)
        - If tag is provided, saves to <root>/{tag}/ (overrides the above)
        Returns the directory written, or None if EMA not available.
        """
        if not hasattr(self, "ema"):
            if self.accelerator.is_main_process:
                print("[save_ema_checkpoint] Skipped: self.ema not initialized.")
            return None

        # Choose output dir name
        if tag is not None:
            out_dir = os.path.join(root_dir, tag)
        else:
            out_dir = (
                os.path.join(root_dir, f"ema_best_step{step}")
                if keep_history else
                os.path.join(root_dir, "ema_best")
            )
        tmp_dir = out_dir + ".tmp"
        os.makedirs(tmp_dir, exist_ok=True)

        self.accelerator.wait_for_everyone()

        # swap EMA weights into the wrapped model only while saving
        ctx = self.ema.apply_to(self.model)
        with ctx:
            state_dict = self.accelerator.get_state_dict(self.model)

            if self.accelerator.is_main_process:
                base = self.accelerator.unwrap_model(self.model)
                base.save_pretrained(
                    tmp_dir,
                    is_main_process=True,
                    save_function=self.accelerator.save,
                    state_dict=state_dict,
                )
                # tiny metadata (no optimizer/scheduler for EMA)
                meta = {"epoch": int(epoch), "step": int(step)}
                if ema_val is not None:
                    meta["ema_val_loss"] = float(ema_val)
                torch.save(meta, os.path.join(tmp_dir, "meta.pt"))

                # atomic-ish replace for the non-history case
                if not keep_history and os.path.exists(out_dir):
                    shutil.rmtree(out_dir)
                os.rename(tmp_dir, out_dir)

                with open(os.path.join(root_dir, "BEST_EMA.txt"), "w") as f:
                    f.write(out_dir)

        self.accelerator.wait_for_everyone()
        return out_dir
     
    def _save_experiment_config(self, experiment_dict, directory):
        with open(os.path.join(directory, "experiment_config.yml"), "w") as f:
            yaml.dump(experiment_dict, f, default_flow_style=False)

    # def save_model(self, directory, epoch=None, loss=None, use_ema=False):
    
    #     self.accelerator.wait_for_everyone()

     
    #     ctx = self.ema.apply_to(self.model) if (use_ema and hasattr(self, "ema")) else contextlib.nullcontext()
    #     with ctx:
    #         # 3) Gather a FULL (global) state_dict regardless of sharding
    #         state_dict = self.accelerator.get_state_dict(self.model)

    #         # 4) Only main process writes files
    #         if self.accelerator.is_main_process:
    #             base = self.accelerator.unwrap_model(self.model)
    #             base.save_pretrained(
    #                 directory,
    #                 is_main_process=True,
    #                 save_function=self.accelerator.save,
    #                 state_dict=state_dict,   # ensure we save the consolidated weights
    #             )
    #             # Save optimizer/scheduler metadata once
    #             torch.save(
    #                 {
    #                     "optimizer": self.optimizer.state_dict(),
    #                     "scheduler": self.scheduler.state_dict(),
    #                     "epoch": epoch,
    #                     "loss": loss,
    #                 },
    #                 os.path.join(directory, f"checkpoint_{'ema' if use_ema else ''}.pt"),
    #             )

    #     # 5) (Optional) another barrier so non-main ranks don’t race ahead and touch the folder
    #     self.accelerator.wait_for_everyone()


#     def save_model(self, directory, epoch=None, loss=None):
#         unwrapped_model = self.accelerator.unwrap_model(self.model)
#         unwrapped_model.save_pretrained(
#         directory,
#         is_main_process=self.accelerator.is_main_process,
#         save_function=self.accelerator.save,
# )   
#         checkpoint = {
#             'optimizer': self.accelerator.unwrap_model(self.optimizer).state_dict(),
#             'scheduler': self.accelerator.unwrap_model(self.scheduler).state_dict(),
#             'epoch': epoch,
#             'loss': loss,
#         }
#         torch.save(checkpoint, os.path.join(directory, 'checkpoint.pt'))

class TrainerVQ(BaseTrainer):
    def __init__(self, args, accelerator, len_dataloader=None):
        self.args = args
        self.model_config = init_configure_model(args)
        super().__init__(model=self._init_model(), args=args, accelerator=accelerator, len_dataloader=len_dataloader)

    def run_train(self, train_dataloader, val_datalaoder, experiment_dict, directory):
        best_loss = float('inf')
        beta_recons = self.args.beta_recons
        for epoch in range(self.args.num_epochs if not self.args.test_pipeline else TEST_LEGNTH):
            if self.accelerator.is_main_process:
                print(f"Epoch {epoch+1}/{self.args.num_epochs}")    
            epoch_loss = 0.0
            epoch_recon_loss = 0.0
            epoch_val_loss = 0.0

            for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Training"):
                # Important before starting one forward pass
                self.optimizer.zero_grad()
                batch = batch.contiguous()
                if self.args.test_pipeline and i > TEST_LEGNTH:
                    break
                if i == 0 and epoch == 0 and self.accelerator.is_main_process:
                    print(f"Batch shape: {batch.shape}")
            
                ## Encoding Step
                if i == 0 and epoch == 0 and self.accelerator.is_main_process:
                    latents = self.model.encode(batch, return_dict=True).latents
                    experiment_dict["input_shape"] = list(batch.shape[1:])
                    experiment_dict["latent_shape"] = list(latents.shape[1:])
                    self._save_experiment_config(experiment_dict, directory)
                    print(f"Batch shape: {batch.shape}")
                    print(f"Latent sample shape: {latents.shape}")
                    out = self.model.decode(latents, return_dict=True)

                else:
                    out = self.model(batch, return_dict=True)
                self.accelerator.wait_for_everyone()
                loss_i  = out.commit_loss
                recons = out.sample
                # Loss Function Computation
                recon_loss_i = self.recons_loss(recons, batch)
                loss_i = beta_recons * recon_loss_i + loss_i
                self.accelerator.backward(loss_i)
                self.optimizer.step()
                self.scheduler.step()
                
                # Track metrics
                epoch_loss += loss_i.item()

                epoch_recon_loss += recon_loss_i.item()
                if self.accelerator.is_main_process:
                    tqdm.write(f"Epoch {epoch+1} - Batch {i+1}/{len(train_dataloader)} - Loss: {loss_i.item():.4f}")
                del recon_loss_i, recons

                self.ema.update(self.model)

                
                # Step optimizer after accumulating gradients
                self.optimizer.zero_grad()
            # Update epoch metrics with batch averages
            epoch_loss /= len(train_dataloader)
            epoch_recon_loss /= len(train_dataloader)
            self.current_epoch += 1
            
            print(f"Epoch {epoch+1}, Loss: {epoch_loss}")
            self.accelerator.log({"epoch": epoch+1, "loss": epoch_loss, "recon_loss": epoch_recon_loss})

            # Saving Best model
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                print(f"New best loss: {best_loss}")
                self.save_model(directory)
            self.accelerator.wait_for_everyone()
    
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

    def run_train(self, train_dataloader, experiment_dict, directory):
        self.model.train()
        best_loss = float('inf')
        beta_recons = self.args.beta_recons

        for epoch in range(self.args.num_epochs if not self.args.test_pipeline else TEST_LEGNTH):
            if self.accelerator.is_main_process:
                print(f"Epoch {epoch+1}/{self.args.num_epochs}")     
            epoch_loss = 0.0
            epoch_kl_loss = 0.0
            epoch_recon_loss = 0.0

            for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Training"):
                # Important before starting one forward pass
                self.optimizer.zero_grad()
                batch = batch.contiguous()
                if self.args.test_pipeline and i > TEST_LEGNTH:
                    break
                if i == 0 and epoch == 0 and self.accelerator.is_main_process:
                    print(f"Batch shape: {batch.shape}")
            
                ## Encoding Step
                posterior = self.model.encode(batch).latent_dist
                mu_posterior = posterior.mean
                logvar_posterior = posterior.logvar
                
                # Decoding step
                posterior_sample = posterior.sample()
                if i == 0 and epoch == 0 and self.accelerator.is_main_process:
                    experiment_dict["input_shape"] = list(batch.shape[1:])
                    experiment_dict["latent_shape"] = list(posterior_sample.shape[1:])
                    self._save_experiment_config(experiment_dict, directory)

                self.accelerator.wait_for_everyone()
                recon_i = self.model.decode(posterior_sample).sample
                
                # Loss Function Computation
                kl_loss_i = -0.5 * torch.sum(1 + logvar_posterior - mu_posterior.pow(2) - torch.exp(logvar_posterior))
                kl_loss_i /= batch.size(0)
                kl_loss_i = self.annealer(kl_loss_i) if self.use_annealing else kl_loss_i
                recon_loss_i = self.recons_loss(recon_i, batch)
                loss_i = beta_recons * recon_loss_i + kl_loss_i 
        
                self.accelerator.backward(loss_i)
                
                # Step optimizer after accumulating gradients
                self.optimizer.step()
                self.scheduler.step()
                del recon_i, posterior_sample, mu_posterior, logvar_posterior

                # Track metrics
                epoch_loss += loss_i.item()
                epoch_kl_loss += kl_loss_i.item()
                epoch_recon_loss += recon_loss_i.item()
                tqdm.write(f"Epoch {epoch + 1} - Batch {i+1}/{len(train_dataloader)} - Loss: {loss_i.item():.4f}")
                
                
            # Update epoch metrics with batch averages
            epoch_loss /= len(train_dataloader)
            epoch_kl_loss /= len(train_dataloader)
            epoch_recon_loss /= len(train_dataloader)
            self.current_epoch += 1
            

            if self.accelerator.is_main_process:
                print(f"Epoch {epoch+1}, Loss: {epoch_loss}")
            self.accelerator.log({"epoch": epoch+1, "loss": epoch_loss, "recon_loss": epoch_recon_loss, "kl_loss": epoch_kl_loss})

            # Saving Best model
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                if self.accelerator.is_main_process:
                    print(f"New best loss: {best_loss}")
                self.save_model(directory)
            self.accelerator.wait_for_everyone()

    def _init_model(self):
        assert self.args.model_name.startswith("vae"), "Model name must start with 'vae' for VAETrainer"
        model = MODELS[self.args.model_name](**self.model_config)
        return model

class TrainerDiffusionNonVAE(BaseTrainer):
    def __init__(self, args, accelerator, len_train_train_dataloader=None, input_shape=None):
        assert input_shape is not None, "input_shape must be provided"
        super().__init__(model=init_configure_diffusion(
            vit_size=args.vit_size,
            patch_size=args.patch_size,
            input_shape=input_shape[-1] # Assuming input shape is (1, height, width)
        ), args=args, accelerator=accelerator, len_dataloader=len_train_train_dataloader)
        
        ema_kwargs = dict() # TODO: Fix this line of code

        if self.is_main:
            self.ema_model = EMA(
                self.unwrap(self.model),
                forward_method_names = ('sample',),
                **ema_kwargs
            )
            # self.ema_model = self.accelerator.prepare(self.ema_model)
            self.ema_model.to(self.accelerator.device)
            
        self.accelerator.wait_for_everyone()
        

    @staticmethod
    def unwrap(model):
        return model.module if hasattr(model, "module") else model
    
    def run_train(self, train_dataloader, experiment_dict, directory):

        best_loss = float('inf')

        self.model.train()
        for epoch in range(self.args.diff_epochs if not self.args.test_pipeline else TEST_LEGNTH):
            if self.accelerator.is_main_process:
                print(f"Epoch {epoch+1}/{self.args.diff_epochs}")     
            epoch_loss = 0.0
            for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Training"):          
                self.optimizer.zero_grad()
                assert batch.shape[-1] == self.image_shape[-1], f"Batch shape {batch.shape} does not match expected shape {self.image_shape}"
                # batch = batch.contiguous()
                # batch = batch * 2 -1  # Normalize to [-1, 1]
                loss_i = self.model(batch)
                if i == 0 and epoch == 0 and self.accelerator.is_main_process:
                    self._save_experiment_config(experiment_dict, directory)
                
                self.accelerator.backward(loss_i)

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                if self.is_main:
                    self.unwrap(self.ema_model).update()
                # self.accelerator.wait_for_everyone()
                # self.unwrap(self.ema_model).update()
                self.accelerator.wait_for_everyone()
                
                epoch_loss += loss_i.item()
            epoch_loss /= len(train_dataloader)
            if self.accelerator.is_main_process:
                print(f"Epoch {epoch+1}, Loss: {epoch_loss}")
            self.accelerator.log({"epoch": epoch+1, "loss": epoch_loss})
            self.current_epoch +=1
            # Saving Best model
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                if self.accelerator.is_main_process:
                    print(f"New best loss: {best_loss}")
                self.save(directory)

            self.accelerator.wait_for_everyone()
    
        print('training complete')

    def get_diff_model(self):
        """
        Returns the diffusion model.
        """
        return self.unwrap(self.model)
    
    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def save_diff(self, path):
        if not self.is_main:
            return

        save_package = dict(
            model = self.accelerator.unwrap_model(self.model).state_dict(),
            ema_model = self.ema_model.state_dict(),
            optimizer = self.accelerator.unwrap_model(self.optimizer).state_dict(),
            scheduler = self.accelerator.unwrap_model(self.scheduler).state_dict(),
            epoch = self.current_epoch,  # Save current epoch
        )

        torch.save(save_package, os.path.join(path, f'checkpoint.pt'))

    def save(self, path):
        if not self.is_main:
            return
    
        self.save_diff(path)
        self.save_model()

    def load_weights(self, directory):
        """
        Load the model weights from the specified directory.
        """
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory {directory} does not exist.")
        
        checkpoint_path = os.path.join(directory, "checkpoint.pt")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file {checkpoint_path} does not exist.")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.accelerator.device)
        self.model = self.accelerator.unwrap_model(self.model)
        self.optimizer = self.accelerator.unwrap_model(self.optimizer)
        self.scheduler = self.accelerator.unwrap_model(self.scheduler)
        self.model.load_state_dict(checkpoint['model'])
        if 'ema_model' in checkpoint: # this is not necessary for now
            self.ema_model.load_state_dict(checkpoint['ema_model'])
        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler'])

        self.model, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.scheduler
        )
        self.accelerator.wait_for_everyone()
        


class TrainerDiffusion(TrainerDiffusionNonVAE):
    def __init__(self, args, vae_model,  accelerator, len_train_train_dataloader=None, input_shape=None):
     
        self.model_vae = self.accelerator.prepare(self.model_vae)  
        self.image_shape = input_shape
        
        # self.image_shape = (1, *image_shape)
        if accelerator.is_main_process:
            encoding_shape = self._get_prediction_shape_image()
        else:
            encoding_shape = None
        self.encoding_shape = accelerator.broadcast_object(encoding_shape, src=0)
        del encoding_shape
        super().__init__(args, accelerator, len_train_train_dataloader=len_train_train_dataloader, input_shape=self.encoding_shape[-1])
              
        ema_kwargs = dict() # TODO: Fix this line of code

        if self.is_main:
            self.ema_model = EMA(
                self.unwrap(self.model),
                forward_method_names = ('sample',),
                **ema_kwargs
            )
            # self.ema_model = self.accelerator.prepare(self.ema_model)
            self.ema_model.to(self.accelerator.device)

        # self.ema_model = EMA(
        #     self.unwrap(self.model),
        #     forward_method_names = ('sample',),
        #     **ema_kwargs
        # )
        # self.ema_model = self.accelerator.prepare(self.ema_model)

            # self.ema_model.to(self.accelerator.device)
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
    
    def run_train(self, train_dataloader, experiment_dict, directory):

        best_loss = float('inf')
        # self.model_vae.eval()
        self.model_vae.train()
        self.model.train()

        for epoch in range(self.args.diff_epochs if not self.args.test_pipeline else TEST_LEGNTH):
            if self.accelerator.is_main_process:
                print(f"Epoch {epoch+1}/{self.args.diff_epochs}")     
            epoch_loss = 0.0
            for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Training"):          
                # Decoding step
                self.model.train()
                latents = self.model_vae.encode(batch).latent_dist.sample()
                # 
                # detach().requires_grad_()
                loss_i = self.model(latents)
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

            # Saving Best model
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                if self.accelerator.is_main_process:
                    print(f"New best loss: {best_loss}")
                self.save(directory)

            self.accelerator.wait_for_everyone()
    
        print('training complete')

    
    def get_vae_model(self):
        """
        Returns the VAE model.
        """
        return self.unwrap(self.model_vae)
    

    def save(self, path):
        super().save(path)
        unwrapped_model = self.accelerator.unwrap_model(self.vae_model)
        unwrapped_model.save_pretrained(
        path,
        is_main_process=self.accelerator.is_main_process,
        save_function=self.accelerator.save)
        del unwrapped_model


    def load_weights(self, directory):
        super().load_weights(directory)
        # 1. Load model weights (after .prepare, so we can unwrap)
        self.model_vae = self.accelerator.unwrap_model(self.model_vae)
        self.vae_model.load_pretrained(directory)
        self.model_vae.eval()
        self.vae_model = self.accelerator.prepare(self.vae_model)
       

    


        
