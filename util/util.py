#%%
import os
import sys

base_path = '/home/ids/xchen-21/FADING'

from einops import rearrange
import math
import re
import numpy as np
import abc
from tqdm.auto import tqdm
import PIL.Image
import PIL.Image as Image
import imageio
from safetensors import safe_open
import torch
import torch.nn.functional as nnf
import torch.nn as nn
from torchvision.transforms.functional import pil_to_tensor

from accelerate.utils import set_seed

from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from diffusers import StableDiffusionPipeline, DDIMScheduler, DDIMInverseScheduler
from diffusers.models.lora import LoRALinearLayer
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

import util.seq_aligner as seq_aligner

from models.unet_2d_condition import UNet2DConditionModelInflated

def get_prompt_embedding(prompt_str, tokenizer, text_encoder, device):
    prompt_input = tokenizer(
        prompt_str,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)
    prompt_embeds = text_encoder(
        prompt_input
    ).last_hidden_state
    return prompt_embeds


def decode_video_latents(latents, vae):
    video_length = latents.shape[2]
    latents = 1 / 0.18215 * latents
    latents = rearrange(latents, "b c f h w -> (b f) c h w")
    video = vae.decode(latents).sample
    video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
    video = (video / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
    video = video.cpu().float()
    return video

#%
def load_from_video_frames(video_path: str, n_sample_frames, sample_frame_rate, starting_frame_id=0):
    is_frames_path = False
    
    # 如果直接读进来所有frame的图
    if os.path.isdir(video_path):
        file_names = os.listdir(video_path)
        is_frames_path = True
    # 如果读进来mp4视频
    else:
        assert os.path.isfile(video_path) and video_path.endswith('.mp4'), "wrong video format"

        frame_images = []
        reader = imageio.get_reader(video_path)
        fps = reader.get_meta_data()['fps']
        for im in reader:
            frame_images.append(im)


    frame_tensors = []
    for i in range(n_sample_frames):
        i_frame = i * sample_frame_rate + starting_frame_id
        if is_frames_path:
            frame_image = Image.open(f'{video_path}/{i_frame:05}.png')
        else:
            frame_image = Image.fromarray(frame_images[i_frame].astype('uint8'))
        frame_image = frame_image.convert("RGB").resize((512, 512), PIL.Image.LANCZOS)
        frame_tensor = pil_to_tensor(frame_image).unsqueeze(0) / 127.5 - 1.0
        frame_tensors.append(frame_tensor)

    video_tensor = torch.concat(frame_tensors).unsqueeze(0)
    return video_tensor
