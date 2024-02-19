'''
this module covers fnctions used to denoise
'''

import sys
sys.path.append('..')
sys.path.append('../src/')

import torch
from tqdm import tqdm
from diffusion.diffuser_training import DDPMScheduler


def denoise(x, model, timesteps, noise_scheduler, device):
    
    x = x.to(device)
    model = model.to(device)
    
    num_sample = x.shape[0]
    
    x_track_ls = []
    x_0_track_ls = []

    for time_step in tqdm(iterable=reversed(range(0, timesteps)), 
                          total=timesteps, dynamic_ncols=False, 
                          desc="Sampling :: ", position=0):

        ts = torch.ones(num_sample, dtype=torch.long, device=device) * time_step
        # z = torch.randn_like(x) if time_step > 1 else torch.zeros_like(x)
        # print(get_gpu_memory())  # in bytes 
        predicted_noise = model(x, ts)
        # print(get_gpu_memory())  # in bytes 

        # detach x and clean the cache
        x = x.detach().cpu()
        predicted_noise = predicted_noise.detach().cpu()
        
        # gc.collect()
        # torch.cuda.empty_cache()

        res = noise_scheduler.step(predicted_noise, time_step, x)

        x, x_orginal = res['prev_sample'],  res['pred_original_sample']

        # only return the smaple and another
        # res = noise_scheduler.step(predicted_noise, time_step, x, return_dict=False)
        # x = res[0]

        # print(get_gpu_memory())  # in bytes 
        #print(process.memory_info().rss)
        # gc.collect()
        # torch.cuda.empty_cache()
        #print(process.memory_info().rss)  # in bytes 
        x_0_track_ls.append(x_orginal.detach().cpu())
        x_track_ls.append(x.detach().cpu())

        x = x.to(device)
        
    return x_track_ls, x_0_track_ls

def generate_diffused_embed(embed_ls, diffuser, timesteps, device, batch_size=4, 
                            num_generated_sample=10, return_all_time_embed=False):

    # for each embed, we are going to generate several diffused embeddings

    noise_scheduler = DDPMScheduler(num_train_timesteps=timesteps)

    # embed in batch
    embed_ls_batch = [embed_ls[i:i+batch_size] for i in range(0, len(embed_ls), batch_size)]
       
    x_track_final_ls, x_0_track_final_ls = [], []

    for _ in range(num_generated_sample):

        x_track_ls_ls, x_0_track_ls_ls = [], []

        for i in range(len(embed_ls_batch)):

            x_diff_input = embed_ls_batch[i]

            # x_diff_input = torch.concat(x_diff_input, dim=0)
            x_track_ls, x_0_track_ls = denoise(x_diff_input, diffuser, timesteps, noise_scheduler, device=device)

            x_track_ls = [x.unsqueeze(0) for x in x_track_ls]
            x_0_track_ls = [x.unsqueeze(0) for x in x_0_track_ls]

            x_track_ls = torch.concat(x_track_ls, dim=0) # t * sub_bz * dim
            x_0_track_ls = torch.concat(x_0_track_ls, dim=0)

            x_track_ls_ls.append(x_track_ls) # t * sub_bz * dim
            x_0_track_ls_ls.append(x_0_track_ls)
        
        # get the full batch embed
        x_track_ls_ls = torch.concat(x_track_ls_ls, dim=1) # t * bz * dim
        x_0_track_ls_ls = torch.concat(x_0_track_ls_ls, dim=1) # t * bz * dim

        x_track_final_ls.append(x_track_ls_ls.unsqueeze(dim=0)) # 1 * t * bz * dim
        x_0_track_final_ls.append(x_0_track_ls_ls.unsqueeze(dim=0)) # 1 * t * bz * dim

    x_track_final_ls = torch.concat(x_track_final_ls, dim=0) # num_generated_sample * t * bz * dim
    x_0_track_final_ls = torch.concat(x_0_track_final_ls, dim=0) # num_generated_sample * t * bz * dim

    # get the embed after the full diffusion
    x_track_time_aft_diff = x_track_final_ls[:, -1, :, :]

    if return_all_time_embed:
        return x_track_final_ls, x_0_track_final_ls

    return x_track_time_aft_diff


def generate_text(x, models, tokenizer, max_length, device, sub_batch, **generate_kwargs):

    # create a subbatch
    x_list = x
    if sub_batch:
        x_list = [x[i:i+sub_batch] for i in range(0, len(x), sub_batch)]

    output_text_ls = []

    for x_latent in x_list:
        # diffuse the sample in the embedding space
        # x_latent = x
        if len(x_latent.shape) == 2:
            x_latent = x.reshape(-1, 4, 768)

        # encoded vector
        x_latent = x_latent.to(device)
        output = models.decoder.to(device)(x_latent)


        # generate dummy inputs and attention with the same length as cent
        inputs = torch.ones(output.shape, dtype=torch.int).to(device)

        sent_outputs = models.model.generate(
            input_ids=inputs,
            encoder_outputs={0: output, }, # in order to use other decoder  strategy
            max_length=max_length, **generate_kwargs)

        outputs_text = tokenizer.batch_decode(
            sent_outputs.detach().cpu(), skip_special_tokens=True)

        output_text_ls.extend(outputs_text)
    
    return output_text_ls