import sys
sys.path.append('..')
sys.path.append('../src/diffusion')

from diffusion.diffusion import Autoencoder, UNet, UNetConv
from dataclasses import dataclass
import torch
import math
import torch
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
#Step 3 optimize reverse sampling
from accelerate import Accelerator
from huggingface_hub import HfFolder, Repository, whoami
import argparse
from tqdm.auto import tqdm
from pathlib import Path
import os
from accelerate import notebook_launcher
import torch.nn.functional as F

class NumpyArrayDataset(Dataset):
    #This class is used to upload the data
    def __init__(self, numpy_array):
        self.data = numpy_array
        
        self.shape = self.data.shape[1]
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # Initialize accelerator and tensorboard logging
    print(os.path.join(config.output_dir, "logs"))
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps, 
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs")
       
    )
    if accelerator.is_main_process:
        accelerator.init_trackers("train_example")
    
    # Prepare everything
    # There is no specific order to remember, you just need to unpack the 
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bs,), device=clean_images.device).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            
            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps)
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1
    return model



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_batch_size", type=int, default=10)
    parser.add_argument("--eval_batch_size", type=int, default=10)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--mixed_precision", type=str, default='fp16')
    parser.add_argument("--model_name", type=str, default='VAE')
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--embedding_input", type=str, default='test')
    parser.add_argument("--output_dir", type=str, default='.')

    args, _ = parser.parse_known_args()
    
    if args.embedding_input == 'test':
        generated_embeddings =  torch.ones(100000,  768 * 2) #The shape (# of samples, # of dimensions)
    else:
        generated_embeddings =  torch.tensor(torch.load(args.embedding_input))

    if args.model_name == 'VAE':
        model = Autoencoder(generated_embeddings.shape[1], args.hidden_dim)
    elif args.model_name == 'UNet':
        model = UNet(generated_embeddings.shape[1], args.hidden_dim)
    elif args.model_name == 'UNet_Conv':
        model = UNetConv(generated_embeddings.shape[2], generated_embeddings.shape[1], generated_embeddings.shape[1])

    

    dataset = NumpyArrayDataset(generated_embeddings)
    # create a dataloader for the dataset
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    #generate argparse argument based on TrainingConfig
    class TrainingConfig:
        image_size =  generated_embeddings.shape[1]  # based on generated embeddings
        train_batch_size = args.train_batch_size
        eval_batch_size = args.eval_batch_size  # how many images to sample during evaluation
        num_epochs = args.num_epochs
        gradient_accumulation_steps = args.gradient_accumulation_steps
        learning_rate = args.learning_rate
        lr_warmup_steps = args.lr_warmup_steps
        mixed_precision = args.mixed_precision  # `no` for float32, `fp16` for automatic mixed precision
        overwrite_output_dir = True  # overwrite the old model when re-running the notebook
        seed = 0,
        output_dir = args.output_dir


    config = TrainingConfig()


    noise_scheduler = DDPMScheduler(num_train_timesteps=100)
    # We still have to define 
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    #Learning rate scheduler
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(dataloader) * config.num_epochs),
    )

    arg = (config, model, noise_scheduler, optimizer, dataloader, lr_scheduler)

    notebook_launcher(train_loop, arg, num_processes=1)
    #Save pytorch model to dict
    output_dir = args.output_dir
    embedding_input = args.embedding_input
    model_name = args.model_name
    torch.save(model.state_dict(), f'{output_dir}/diffusion_model.pt')
    
    