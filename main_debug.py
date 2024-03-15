# %%
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# %
import sys
import os
base_path = '/home/ids/xchen-21/FADING2'
import argparse
from einops import rearrange
import numpy as np
import abc
from tqdm.auto import tqdm

from accelerate.utils import set_seed

from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from diffusers import StableDiffusionPipeline, DDIMScheduler, DDIMInverseScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.models.lora import LoRALinearLayer
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from util.util import *

from models.unet_2d_condition import UNet2DConditionModelInflated
import cv2
# from insightface.app import FaceAnalysis


weight_dtype = torch.float32
device = torch.device('cuda')

seed = 2024

generator = torch.Generator(device="cuda")
generator.manual_seed(seed)
set_seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# pretrained_model_path = "runwayml/stable-diffusion-v1-5"
pretrained_model_path = "findnitai/FaceGen"
# pretrained_model_path = "SG161222/Realistic_Vision_V6.0_B1_noVAE"

tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")

text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
text_encoder.requires_grad_(False)
text_encoder.to(device=device, dtype=weight_dtype)

vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
vae.requires_grad_(False)
vae.to(device=device, dtype=weight_dtype)

unet = UNet2DConditionModelInflated.from_pretrained(
    pretrained_model_path, subfolder="unet")
unet.requires_grad_(False)
unet.to(device=device, dtype=weight_dtype)
unet.enable_gradient_checkpointing()

ddim_steps = 50
forward_scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
inverse_scheduler = DDIMInverseScheduler.from_config(forward_scheduler.config)

forward_scheduler.set_timesteps(ddim_steps)
inverse_scheduler.set_timesteps(ddim_steps)

#%% load input
video_path = "/home/ids/xchen-21/FADING/data/CelebV-HQ/downloaded_celebvhq/processed/0s1UUn9aSSw_1.mp4"
video_name = video_path.split('/')[-1][:-4]

n_sample_frames = 1
sample_frame_rate = 10
starting_frame_id = 0
# starting_frame_id = 40

video_tensor = load_from_video_frames(video_path,
                                      n_sample_frames,
                                      sample_frame_rate,
                                      starting_frame_id)
pixel_values = video_tensor.to(device=device, dtype=weight_dtype)

n_frames = pixel_values.shape[1]
pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
# use vae mean??
input_latents = vae.encode(pixel_values).latent_dist.sample()
input_latents = rearrange(input_latents, "(b f) c h w -> b c f h w", f=n_frames)
input_latents = input_latents * 0.18215
print(f"input processed, init latent shape {input_latents.shape}")


#%% set attn processor:
from attention_processors import *

attnStore = AttnStore(ddim_steps, device, weight_dtype, tokenizer)
set_attn_processor_with_store(unet, attnStore)
print("Attention processors set to UNet.")

# %% DDIM inversion
prompt_src = "photo of a woman"
emb_src = get_prompt_embedding(prompt_src, tokenizer, text_encoder, device)

attnStore.set_mode('store')

print(f"inversion prompt: {prompt_src}")
ddim_intermediate_latents = []
latents = input_latents.detach().clone()
for i, t in enumerate(tqdm(inverse_scheduler.timesteps.tolist(),
                           desc="DDIM Inversion", leave=False)):
    # if i == 2: break  ##
    attnStore.set_i_t(i)

    noise_pred = unet(latents, t, encoder_hidden_states=emb_src).sample
    latents = inverse_scheduler.step(noise_pred, t, latents).prev_sample
    ddim_intermediate_latents.append(latents)

ddim_inverted_latents = latents.detach().clone()

attnStore.set_mode(None)
print("DDIM done")


# %% editing
prompt_tgt = "photo of a woman with black hair"
i_mask_word = ["hair"]
prompts = [prompt_src, prompt_tgt]

words_encode = [tokenizer.decode([item]) for item in tokenizer.encode(prompt_tgt)]
i_mask_token = [words_encode.index(word) for word in i_mask_word]

emb_tgt = get_prompt_embedding(prompt_tgt, tokenizer, text_encoder, device)

prompt_nega = prompt_src
emb_nega = get_prompt_embedding(prompt_nega, tokenizer, text_encoder, device)

guidance_scale = 5.  ##

attnStore.set_mode('replace'); self_replace = 0.8 ##

cross_replace = 1.
self_replace_res = 64

self_attention_blend = len(i_mask_token) > 0
attnStore.set_replace_params(
    cross_replace=cross_replace,
    self_replace=self_replace,
    self_replace_res=self_replace_res,
    self_attention_blend=self_attention_blend,  ##
    prompts=prompts,
    i_mask_token=i_mask_token,
)

print(f"editing prompt: {prompt_tgt}")
latents = ddim_inverted_latents.detach().clone()
for i, t in enumerate(tqdm(forward_scheduler.timesteps.tolist(),
                           desc="Editing", leave=False)):
    i_ddim_inversion = ddim_steps - i - 1

    latents_with_grad = latents.detach().clone()
    if attnStore.mode == 'guidance':
        latents_with_grad.requires_grad = True

    # target ppt 作为positive prompt/cond emb
    attnStore.set_i_t(i_ddim_inversion)
    attnStore.set_is_recon(False)

    attnStore.set_cross_attn_mask(i_ddim_inversion, threshold=0.3, binary=False)

    # if i % 10 == 0 and i > 0:
    #     pass
    #     attn_mask = attnStore.cross_attn_mask.cpu().numpy()
    #     attn_mask = attn_mask.reshape(attn_mask.shape[0] * attn_mask.shape[1], -1)
    #     mydisplay(attn_mask, str(i_ddim_inversion))

    posi_noise_pred = unet(latents_with_grad, t, encoder_hidden_states=emb_tgt).sample

    # source ppt 作为negative prompt/uncond emb
    attnStore.set_i_t(i_ddim_inversion)
    attnStore.set_is_recon(True)

    # nega_noise_pred = unet(latents_with_grad, t, encoder_hidden_states=emb_src).sample
    nega_noise_pred = unet(latents_with_grad, t, encoder_hidden_states=emb_nega).sample

    noise_pred = nega_noise_pred + guidance_scale * (posi_noise_pred - nega_noise_pred)

    # 这里应该是latents_with_grad!!!，不然会memory loss
    latents = forward_scheduler.step(noise_pred, t, latents_with_grad, generator=generator).prev_sample
    # latents = forward_scheduler.step(noise_pred, t, latents, generator=generator).prev_sample

    # local blend with cross attention mask
    if attnStore.self_attention_blend and i > 40:
        attnStore.set_cross_attn_mask(i_ddim_inversion, threshold=0.2, binary=True)
        attn_mask = attnStore.cross_attn_mask.gt(0.2).to(dtype=weight_dtype).unsqueeze(0).unsqueeze(0)
        latents = latents*attn_mask + ddim_intermediate_latents[i_ddim_inversion]*(1-attn_mask)

# mydisplay(attn_mask.squeeze().cpu().numpy())

edited_video = decode_video_latents(latents.detach(), vae)
attnStore.set_mode(None)
print("Editing done.")


frames = []
for i in range(edited_video.shape[2]):
    frame = edited_video[:, :, i, :, :].squeeze().permute(1, 2, 0)
    frame = (frame * 255).numpy().astype(np.uint8)
    i_frame = i * sample_frame_rate + starting_frame_id
    Image.fromarray(frame).save(f'output/tmp.png')
