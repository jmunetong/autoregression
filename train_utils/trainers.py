import os

import yaml
import torch
from tqdm import tqdm
from .annealing import Annealer

from typing import List

import math
from pathlib import Path

from accelerate import Accelerator
from ema_pytorch import EMA

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from torchvision.utils import save_image
from torch.nn import Module



TEST_LEGNTH = 1

class BaseTrainer():
    def __init__(self, args, model, optimizer, scheduler, accelerator,  recons_loss):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.accelerator = accelerator
        self.recons_loss = recons_loss

    def run_trainer(self, data_loader, experiment_dict, directory):
        """
        Run the training loop for the model.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def save_model(self, directory):
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(
        directory,
        is_main_process=self.accelerator.is_main_process,
        save_function=self.accelerator.save,
)
        
    def _save_experiment_config(self, experiment_dict, directory):
        with open(os.path.join(directory, "experiment_config.yml"), "w") as f:
            yaml.dump(experiment_dict, f, default_flow_style=False)



class TrainerVQ(BaseTrainer):
    def __init__(self, args, model, optimizer, scheduler, accelerator, recons_loss):
        super().__init__(args, model, optimizer, scheduler, accelerator, recons_loss)
    
    def run_train(self, data_loader, experiment_dict, directory):
        best_loss = float('inf')
        beta_recons = self.args.beta_recons
        for epoch in range(self.args.num_epochs if not self.args.test_pipeline else TEST_LEGNTH):
            if self.accelerator.is_main_process:
                print(f"Epoch {epoch+1}/{self.args.num_epochs}")    
            epoch_loss = 0.0
            epoch_recon_loss = 0.0

            for i, batch in tqdm(enumerate(data_loader), total=len(data_loader), desc="Training"):
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
                    tqdm.write(f"Epoch {epoch+1} - Batch {i+1}/{len(data_loader)} - Loss: {loss_i.item():.4f}")
                del recon_loss_i, recons
                
                # Step optimizer after accumulating gradients
                
                
            # Update epoch metrics with batch averages
            epoch_loss /= len(data_loader)
            epoch_recon_loss /= len(data_loader)
            
            print(f"Epoch {epoch+1}, Loss: {epoch_loss}")
            self.accelerator.log({"epoch": epoch+1, "loss": epoch_loss, "recon_loss": epoch_recon_loss})

            # Saving Best model
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                print(f"New best loss: {best_loss}")
                self.save_model(directory)
            self.accelerator.wait_for_everyone()


class TrainerVAE(BaseTrainer):
    def __init__(self, args, model, optimizer, scheduler, accelerator, recons_loss):
        super().__init__(args, model, optimizer, scheduler, accelerator, recons_loss)
        self.use_annealing = args.use_annealing
        if self.use_annealing:
            total_steps = args.num_epochs
            shape = args.annealing_shape
            baseline = 0.0
            cyclical = False
            disable = False

            self.annealer = Annealer(total_steps, shape, baseline, cyclical)

       
    def run_train(self, data_loader, experiment_dict, directory):
        best_loss = float('inf')
        beta_recons = self.args.beta_recons

        for epoch in range(self.args.num_epochs if not self.args.test_pipeline else TEST_LEGNTH):
            if self.accelerator.is_main_process:
                print(f"Epoch {epoch+1}/{self.args.num_epochs}")     
            epoch_loss = 0.0
            epoch_kl_loss = 0.0
            epoch_recon_loss = 0.0

            for i, batch in tqdm(enumerate(data_loader), total=len(data_loader), desc="Training"):
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
                tqdm.write(f"Epoch {epoch + 1} - Batch {i+1}/{len(data_loader)} - Loss: {loss_i.item():.4f}")
                
                
            # Update epoch metrics with batch averages
            epoch_loss /= len(data_loader)
            epoch_kl_loss /= len(data_loader)
            epoch_recon_loss /= len(data_loader)
            

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

class TrainerDiffusion(BaseTrainer):
    def __init__(self, args, model,diff_model, optimizer, scheduler, accelerator,image_shape):
        super().__init__(args, model, optimizer, scheduler, accelerator, recons_loss=None)
        self.model_vae = self.unwrap(model)
        model_dim = dict(
        dim = 1024,
        depth = 12,
        heads = 12) #TODO: Add experiment parameters for these values
        self.patch_size = 8 # TODO: Add experiment parameters for this value
        self.image_shape = (1, *image_shape)
        self.encoding_shape = None
        self._get_prediction_shape_image()
        self.model = self.accelerator.prepare(diff_model(model=model_dim, image_size = self.encoding_shape[-1], patch_size = self.patch_size))
        
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
        self.encoding_shape = out.shape[1:]
    
    @staticmethod
    def unwrap(model):
        return model.module if hasattr(model, "module") else model
    
    def run_train(self, data_loader, experiment_dict, directory):

        best_loss = float('inf')
        self.model_vae.eval()
        self.model.train()

        for epoch in range(self.args.diff_epochs if not self.args.test_pipeline else TEST_LEGNTH):
            if self.accelerator.is_main_process:
                print(f"Epoch {epoch+1}/{self.args.diff_epochs}")     
            epoch_loss = 0.0
            for i, batch in tqdm(enumerate(data_loader), total=len(data_loader), desc="Training"):          
                # Decoding step
                self.model.train()
                latents = self.model_vae.encode(batch).latent_dist.sample().detach().requires_grad_()
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

    def get_diff_model(self):
        """
        Returns the diffusion model.
        """
        return self.unwrap(self.model)
    
    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def save(self, path):
        if not self.is_main:
            return

        save_package = dict(
            model = self.accelerator.unwrap_model(self.model).state_dict(),
            ema_model = self.ema_model.state_dict(),
            optimizer = self.accelerator.unwrap_model(self.optimizer).state_dict(),
        )

        torch.save(save_package, os.path.join(path, f'checkpoint.pt'))


        

class TrainerDiffusionNonVAE(BaseTrainer):
    def __init__(self, args, model,diff_model, optimizer, scheduler, accelerator,image_shape):
        super().__init__(args, model, optimizer, scheduler, accelerator, recons_loss=None)
        self.model_vae = self.unwrap(model)
        model_dim = dict(
        dim = 512,
        depth = 3,
        heads = 1) #TODO: Add experiment parameters for these values
        self.patch_size = 8 # TODO: Add experiment parameters for this value
        self.image_shape = (1, *image_shape)
        self.encoding_shape = None
        self._get_prediction_shape_image()
        self.model = self.accelerator.prepare(diff_model(model=model_dim, image_size = self.encoding_shape[-1], patch_size = self.patch_size))
        
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
        self.encoding_shape = out.shape[1:]
    
    @staticmethod
    def unwrap(model):
        return model.module if hasattr(model, "module") else model
    
    def run_train(self, data_loader, experiment_dict, directory):

        best_loss = float('inf')
        self.model_vae.eval()
        self.model.train()

        for epoch in range(self.args.diff_epochs if not self.args.test_pipeline else TEST_LEGNTH):
            if self.accelerator.is_main_process:
                print(f"Epoch {epoch+1}/{self.args.diff_epochs}")     
            epoch_loss = 0.0
            for i, batch in tqdm(enumerate(data_loader), total=len(data_loader), desc="Training"):          
                # Decoding step
                self.model.train()
                latents = self.model_vae.encode(batch).latent_dist.sample().detach().requires_grad_()
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
            epoch_loss /= len(data_loader)
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

    def get_diff_model(self):
        """
        Returns the diffusion model.
        """
        return self.unwrap(self.model)
    
    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def save(self, path):
        if not self.is_main:
            return

        save_package = dict(
            model = self.accelerator.unwrap_model(self.model).state_dict(),
            ema_model = self.ema_model.state_dict(),
            optimizer = self.accelerator.unwrap_model(self.optimizer).state_dict(),
        )

        torch.save(save_package, os.path.join(path, f'checkpoint.pt'))


        
