import os

import torch
from tqdm import tqdm

TEST_LEGNTH = 3

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
        # Save the model
        # torch.save(self.model.state_dict(), os.path.join(directory, f"vq_model.pth"))
        # try:
        #     self.model.save_pretrained(os.path.join(directory, f"vq_model_pretrained"))
        # except Exception as e:
        #     print(f"Error saving feature extractor: {e}")
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        # self.accelerator.save_model(self.model, directory)

        unwrapped_model.save_pretrained(
        directory,
        is_main_process=self.accelerator.is_main_process,
        save_function=self.accelerator.save,
)



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
                    print(f"Batch shape: {batch.shape}")
                    print(f"Latent sample shape: {latents.shape}")
                    out = self.model.decode(latents, return_dict=True)

                else:
                    out = self.model(batch, return_dict=True)
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


class TrainerVAE(BaseTrainer):
    def __init__(self, args, model, optimizer, scheduler, accelerator, recons_loss):
        super().__init__(args, model, optimizer, scheduler, accelerator, recons_loss)
       
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
                    print(f"Batch shape: {batch.shape}")
                    print(f"Posterior sample shape: {posterior_sample.shape}")
                
                recon_i = self.model.decode(posterior_sample).sample
                
                # Loss Function Computation
                kl_loss_i = -0.5 * torch.sum(1 + logvar_posterior - mu_posterior.pow(2) - torch.exp(logvar_posterior))
                kl_loss_i /= batch.size(0)
                recon_loss_i = self.recons_loss(recon_i, batch)
                loss_i = beta_recons * recon_loss_i + kl_loss_i #TODO TEST THIS PARAMETER
                
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
            

            print(f"Epoch {epoch+1}, Loss: {epoch_loss}")
            self.accelerator.log({"epoch": epoch+1, "loss": epoch_loss, "recon_loss": epoch_recon_loss, "kl_loss": epoch_kl_loss})

            # Saving Best model
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                if self.accelerator.is_main_process:
                    print(f"New best loss: {best_loss}")
                self.save_model(directory)

