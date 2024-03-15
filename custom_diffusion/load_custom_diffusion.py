
from einops import rearrange
import math
import re
import numpy as np
import abc
from tqdm.auto import tqdm
import PIL.Image
import PIL.Image as Image
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


#%% load custom diffusion finetuned weights
# model_name = 'base-photo of a woman with <style2> hair'
# model_name = 'base-photo of a woman with <style2> hair-photo of a woman'
# model_name = 'swapattn-photo of a woman with <style2> hair-photo of a woman'
model_name = 'base-photo of a woman with <style2> hair-photo of a woman with <style1> hair'
finetuned_model_path = f'{base_path}/diffusers/examples/custom_diffusion/finetuned_models/DDS/' \
                       f'{model_name}'

finetune_steps = 400

pipeline_useless = StableDiffusionPipeline.from_pretrained(
    pretrained_model_path, safety_checker=None,
).to(device=device, dtype=weight_dtype)

# load modifier_token emb
for modifier_token in re.findall(r'<(.*?)>', model_name):
    pipeline_useless.load_textual_inversion(finetuned_model_path, weight_name=f"<{modifier_token}>-{finetune_steps}.bin")
    print(f"<{modifier_token}> embedding loaded")

tokenizer = pipeline_useless.tokenizer
text_encoder = pipeline_useless.text_encoder
text_encoder.requires_grad_(False)
print('custom text encoder and tokinzer loaded')

# load custom cross attention weights
pipeline_useless.unet.load_attn_procs(finetuned_model_path, weight_name=f"pytorch_custom_diffusion_weights-{finetune_steps}.bin")
attn_custom_processors = pipeline_useless.unet.attn_processors
del pipeline_useless

def set_custom_to_attn(unet, attn_custom_processors):
    def fn_recursive_set_custom_to_attn(name: str, module: torch.nn.Module):
        if hasattr(module, "set_processor"):
            # name: "down_blocks.0.attentions.0.transformer_blocks.0.attn2"
            attn_custom_proc = attn_custom_processors[name+".processor"]

            if name.endswith('attn2'):  # cross attn
                module.to_k.weight.data = attn_custom_proc.to_k_custom_diffusion.weight.data.clone()
                module.to_q.weight.data = attn_custom_proc.to_q_custom_diffusion.weight.data.clone()
                module.to_v.weight.data = attn_custom_proc.to_v_custom_diffusion.weight.data.clone()
                module.to_out[0].weight.data = attn_custom_proc.to_out_custom_diffusion[0].weight.data.clone()
                module.to_out[0].bias.data = attn_custom_proc.to_out_custom_diffusion[0].bias.data.clone()

                print(f"{name} custom weight copied")

        for sub_name, child in module.named_children():
            fn_recursive_set_custom_to_attn(f"{name}.{sub_name}", child)

    for name, module in unet.named_children():
        fn_recursive_set_custom_to_attn(name, module)

set_custom_to_attn(unet, attn_custom_processors)