from typing import Callable, Optional, Union
import xformers
import torch
import torch.nn as nn
from einops import rearrange, repeat
import util.seq_aligner as seq_aligner
from einops import rearrange
import math
import torch
import torch.nn.functional as nnf
import torch.nn as nn
from diffusers.models.attention_processor import Attention
from diffusers.models.lora import LoRALinearLayer


class AttnStore:
    def __init__(self, ddim_steps, device, weight_dtype, tokenizer):

        self.ddim_steps = ddim_steps
        self.device = device
        self.weight_dtype = weight_dtype
        self.tokenizer = tokenizer

        self.init_saved_attns()

        self.self_replace = 0.
        self.cross_replace = 0.

        self.self_replace_res = 64
        self.n_reconstruction_loss = 16

        self.mode = None
        self.is_recon = None

        self.prompts = None
        self.mapper = None
        self.alphas = None
        self.i_mask_token = None
        self.self_attention_blend = None

    def init_saved_attns(self):
        # self.stored_cross_attn = {t:[] for t in forward_scheduler.timesteps.tolist()}
        # self.stored_self_attn = {t:[] for t in forward_scheduler.timesteps.tolist()}
        self.stored_cross_attn_maps = [[] for i_t in range(self.ddim_steps)]
        self.stored_ip_cross_attn_maps = [[] for i_t in range(self.ddim_steps)]

        self.stored_self_attn_qs = [[] for i_t in range(self.ddim_steps)]
        self.stored_self_attn_ks = [[] for i_t in range(self.ddim_steps)]

    def set_replace_params(self, **params_dict):
        for param in params_dict:
            self.__setattr__(param, params_dict[param])
        if "self_replace_res" in params_dict:
            if params_dict["self_replace_res"] in [64,32,16]:
                self.n_reconstruction_loss = 5
            else:
                self.n_reconstruction_loss = 1
        if self.prompts:
            self.mapper, alphas = seq_aligner.get_refinement_mapper(self.prompts, self.tokenizer)
            self.mapper, alphas = self.mapper.to(self.device), alphas.to(self.device)
            self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])

    def set_mode(self, mode: str):
        self.mode = mode
        if mode == 'store':
            self.init_saved_attns()
        if mode == 'replace' or mode == 'guidance':
            assert len(self.stored_cross_attn_maps[-1]) == 16
            self.stored_cross_attn_maps_edit = [[] for i_t in range(self.ddim_steps)]
            self.stored_ip_cross_attn_maps_edit = [[] for i_t in range(self.ddim_steps)]
        if mode == 'guidance':
            self.reconstruction_loss = torch.tensor(0.).to(device=self.device, dtype=self.weight_dtype)

    def set_i_t(self, i_t):
        self.i_t = i_t
        self.i_attn = 0
        if self.mode == 'guidance':
            self.reconstruction_loss = torch.tensor(0.).to(device=self.device, dtype=self.weight_dtype)

    def set_is_recon(self, is_recon: bool):
        self.is_recon = is_recon

    def self_call(self, q, k, attn):
        if self.mode == 'store':
            return self.store_self_attn_qk(q, k)
        if self.mode == 'replace':
            return self.replace_self_attn_qk(q, k)
        if self.mode == 'guidance':
            return self.guidance_self_attn_qk(q, k, attn)
        return q, k

    def cross_call(self, attn, is_ip=False):
        if self.mode == 'store':
            return self.store_cross_attn_map(attn, is_ip)
        if self.mode == 'replace' or self.mode == 'guidance':
            return self.replace_cross_attn_map(attn, is_ip)
        return attn

    def store_cross_attn_map(self, attn, is_ip):
        if not is_ip:  # original cross attention for text prompt
            self.stored_cross_attn_maps[self.i_t].append(attn.cpu())
        else:
            self.stored_ip_cross_attn_maps[self.i_t].append(attn.cpu())
        return attn


    def store_self_attn_qk(self, q, k):
        self.stored_self_attn_qs[self.i_t].append(q.cpu())
        self.stored_self_attn_ks[self.i_t].append(k.cpu())
        return q, k

    def replace_cross_attn_map(self, attn_curr, is_ip):
        if is_ip:
            if not self.is_recon:
                self.stored_ip_cross_attn_maps_edit[self.i_t].append(attn_curr.cpu())
            return attn_curr

        attn_ref = attn_curr
        if self.i_t > self.ddim_steps * (1 - self.cross_replace):
            attn_ref = self.stored_cross_attn_maps[self.i_t][self.i_attn].to(device=self.device, dtype=self.weight_dtype)
            # print(f'cross replaced, {attn_ref.shape}')

        # if not self.is_recon:
        if 1:
            # only do cross attention control for target prompt

            ## p2p attn control here!!
            assert (self.mapper is not None) and (self.alphas is not None), "mapper and alphas not set"
            attn_base = attn_ref  # inversion map
            att_replace = attn_curr

            attn_base_replace = attn_base[:, :, self.mapper.squeeze()]  # inversion map
            attn_ref = attn_base_replace * self.alphas.squeeze(0) + \
                       att_replace * (1 - self.alphas.squeeze(0))

            # # reweight attention
            # per_word_mean = attn_ref[:, :, 1:10].mean(dim=1).mean(dim=-1)
            # key_word_mean = attn_ref[:, :, 11].mean(dim=-1)  # 8
            # attn_ref[:, :, 11] = attn_ref[:, :, 11] / key_word_mean.unsqueeze(1) * per_word_mean.unsqueeze(1) * 5
            # # attn_ref = attn_ref.softmax(dim=-1)

            # store attention maps during edit
            if not self.is_recon:
                self.stored_cross_attn_maps_edit[self.i_t].append(attn_ref.cpu())

        self.i_attn += 1
        return attn_ref

    def set_cross_attn_mask(self, i_t, threshold=0.3, binary=False):
        # if first step or if no mask word, set mask at all zero (all from target)
        if i_t == self.ddim_steps - 1 or len(self.i_mask_token)==0:
            n_frames = self.stored_self_attn_qs[0][0].shape[0]//8
            attn_mask = torch.zeros((n_frames, 64, 64), dtype=torch.float32).to(self.device)
            self.cross_attn_mask = attn_mask
            return

        # use previous timestep (t+1)'s cross attention map as mask
        assert len(self.stored_cross_attn_maps_edit[i_t+1]) == 16
        attn_64_stack = []
        for attn in self.stored_cross_attn_maps_edit[i_t+1]:  # (8f,4096,77)
            attn = attn[:, :, self.i_mask_token].mean(dim=-1)  # (8f,4096)
            attn_for_each_frame = torch.split(attn, 8, dim=0)  # [(8,4096) *f]
            attn_avg_each_frame = [i.mean(dim=0, keepdim=True) for i in attn_for_each_frame]  # [(1,4096) *f]
            attn = torch.cat(attn_avg_each_frame, dim=0)  # (f,4096)

            res = int(math.sqrt(attn.shape[-1]))
            attn = attn.reshape(-1, res, res)  # (f,64,64)

            if res < 64:  # 在这里选特定res的
                attn = nnf.interpolate(attn.unsqueeze(0),
                                       size=(64, 64), mode='nearest').squeeze(0)
            attn_64_stack.append(attn)

        attn_mask = torch.stack(attn_64_stack, dim=0).mean(dim=0, keepdim=False)  # (f,64,64)
        attn_mask = (attn_mask - attn_mask.min()) / (attn_mask.max() - attn_mask.min())  # rescale to (0,1)
        threshold = threshold
        threshold_mask = attn_mask.gt(threshold).to(dtype=self.weight_dtype)
        if not binary:
            attn_mask = attn_mask * threshold_mask
        else:
            attn_mask = threshold_mask

        self.cross_attn_mask = attn_mask.to(self.device)

    def replace_self_attn_qk(self, q_curr, k_curr):
        q_ref, k_ref = q_curr, k_curr
        if self.i_t > self.ddim_steps * (1 - self.self_replace):

            # if replace, only use maps of one resolution to guide
            if q_curr.shape[1] <= self.self_replace_res ** 2:
                q_ref = self.stored_self_attn_qs[self.i_t][self.i_attn].to(device=self.device, dtype=self.weight_dtype)
                k_ref = self.stored_self_attn_ks[self.i_t][self.i_attn].to(device=self.device, dtype=self.weight_dtype)
                # print(f'self qk replaced, q shape {q_ref.shape}, k shape {k_ref.shape}')

                if self.self_attention_blend is True:
                    # self attention blend
                    assert self.cross_attn_mask is not None
                    mask = self.cross_attn_mask   # (f,64,64)
                    n_frames = mask.shape[0]
                    res = int(math.sqrt(q_ref.shape[1]))
                    mask = nnf.interpolate(mask.unsqueeze(0),
                                           size=(res, res), mode='bilinear').squeeze(0)   # (f,res,res)
                    mask = mask.reshape(n_frames, res*res, 1)  # (f, res^2, 1)

                    # repeat 8 times
                    mask = mask.unsqueeze(1).repeat(1,8,1,1)
                    mask = rearrange(mask, "f h n d -> (f h) n d")  # (8f, res^2, 1)

                    # repeat frame times for k
                    mask_repeat_frame = rearrange(mask, "(b f) n d -> b f n d", f=n_frames)
                    mask_repeat_frame = mask_repeat_frame.unsqueeze(2).repeat(1, 1, n_frames, 1, 1)  # (b f f n d)
                    # mask_repeat_frame = mask_repeat_frame.unsqueeze(2).repeat(1, 1, 2, 1, 1)  # (b f f n d)
                    mask_repeat_frame = rearrange(mask_repeat_frame, "b f g n d -> (b f) (g n) d")

                    q_ref = q_curr * mask + q_ref * (1-mask)
                    k_ref = k_curr * mask_repeat_frame + k_ref * (1-mask_repeat_frame)

        return q_ref, k_ref



class LoRAIPAttnProcessor(nn.Module):
    r"""
    Attention processor for IP-Adapater.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        scale (`float`, defaults to 1.0):
            the weight scale of image prompt.
        num_tokens (`int`, defaults to 4 when do ip_adapter_plus it should be 16):
            The context length of the image features.
    """

    def __init__(self, hidden_size, cross_attention_dim=None, rank=4, network_alpha=None,
                 lora_scale=0.0,
                 ip_scale=0.0, num_tokens=0,
                 attnStore: AttnStore = None):

        assert lora_scale == 0, "现在不加lora"
        super().__init__()

        self.attnStore = attnStore

        self.rank = rank
        self.lora_scale = lora_scale

        self.to_q_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
        self.to_k_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_v_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)

        self.to_out_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim

        assert ip_scale == 0 and num_tokens == 0, "现在不加IP adapter"
        self.ip_scale = ip_scale
        self.num_tokens = num_tokens

        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)

    def set_ip_scale(self, ip_scale):
        self.ip_scale = ip_scale

    def __call__(
            self,
            attn: Attention,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            temb: Optional[torch.FloatTensor] = None,
            scale: float = 1.0,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        if self.lora_scale > 0:
            query += self.lora_scale * self.to_q_lora(hidden_states)

        if encoder_hidden_states is None:
            raise ValueError("it should be cross attn here")
            # encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        ## 把text prompt的emb 和 face id 的 emb 分开
        end_pos = encoder_hidden_states.shape[1] - self.num_tokens
        encoder_hidden_states, ip_hidden_states = (
            encoder_hidden_states[:, :end_pos, :],
            encoder_hidden_states[:, end_pos:, :],
        )
        if attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        # original cross attention for text prompt
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if self.lora_scale > 0:
            key += self.lora_scale * self.to_k_lora(encoder_hidden_states)
            value += self.lora_scale * self.to_v_lora(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)

        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        ## store or edit cross attention maps
        assert self.attnStore, "Set self.attnStore first"
        attention_probs = self.attnStore.cross_call(attention_probs)
        ##

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        ## for ip-adapter: image prompt/face id 的另一层cross attn
        if self.ip_scale > 0:
            ip_key = self.to_k_ip(ip_hidden_states)
            ip_value = self.to_v_ip(ip_hidden_states)

            ip_key = attn.head_to_batch_dim(ip_key)
            ip_value = attn.head_to_batch_dim(ip_value)

            ip_attention_probs = attn.get_attention_scores(query, ip_key, None)
            ip_attention_probs = self.attnStore.cross_call(ip_attention_probs, is_ip=True)
            # self.attnStore.attn_map = ip_attention_probs
            ip_hidden_states = torch.bmm(ip_attention_probs, ip_value)
            ip_hidden_states = attn.batch_to_head_dim(ip_hidden_states)

            hidden_states = hidden_states + self.ip_scale * ip_hidden_states
            ##

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        if self.lora_scale > 0:
            hidden_states += self.lora_scale * self.to_out_lora(hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

class LoRAXFormersSpatialTemporalAttnProcessor(nn.Module):
    def __init__(self,
                 hidden_size,
                 cross_attention_dim=None,
                 rank=4,
                 network_alpha=None,
                 lora_scale=0.0,
                 ):
        super().__init__()

        self.attention_op = None
        self.attnStore = None

        self.rank = rank
        self.lora_scale = lora_scale

        self.to_q_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
        self.to_k_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_v_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_out_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)


    def rearrangeKV_spatial_temporal(self, key, value, video_length):
        key = rearrange(key, "(b f) n d -> b f n d", f=video_length)
        key = key.unsqueeze(1).repeat(1, video_length, 1, 1, 1)  # (b f f n d)
        key = rearrange(key, "b f g n d -> (b f) (g n) d")

        value = rearrange(value, "(b f) n d -> b f n d", f=video_length)
        value = value.unsqueeze(1).repeat(1, video_length, 1, 1, 1)  # (b f f n d)
        value = rearrange(value, "b f g n d -> (b f) (g n) d")

        return key, value

    # def rearrangeKV_sparse_causal(self, key, value, video_length):
    #     former_frame_index = torch.arange(video_length) - 1
    #     former_frame_index[0] = 0
    #
    #     key = rearrange(key, "(b f) d c -> b f d c", f=video_length)
    #     key = torch.cat([key[:, [0] * video_length], key[:, former_frame_index]], dim=2)
    #     key = rearrange(key, "b f d c -> (b f) d c")
    #
    #     value = rearrange(value, "(b f) d c -> b f d c", f=video_length)
    #     value = torch.cat([value[:, [0] * video_length], value[:, former_frame_index]], dim=2)
    #     value = rearrange(value, "b f d c -> (b f) d c")
    #
    #     return key, value

    def __call__(
            self,
            attn: Attention,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            temb: Optional[torch.FloatTensor] = None,
            scale: float = 1.0,
            video_length: int = None,
    ) -> torch.FloatTensor:
        residual = hidden_states

        args = ()

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, key_tokens, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        attention_mask = attn.prepare_attention_mask(attention_mask, key_tokens, batch_size)
        if attention_mask is not None:
            # expand our mask's singleton query_tokens dimension:
            #   [batch*heads,            1, key_tokens] ->
            #   [batch*heads, query_tokens, key_tokens]
            # so that it can be added as a bias onto the attention scores that xformers computes:
            #   [batch*heads, query_tokens, key_tokens]
            # we do this explicitly because xformers doesn't broadcast the singleton dimension for us.
            _, query_tokens, _ = hidden_states.shape
            attention_mask = attention_mask.expand(-1, query_tokens, -1)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            raise ValueError("it should be self attn here")
        # elif attn.norm_cross:
        #     encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        query = attn.to_q(hidden_states, *args)
        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        if self.lora_scale > 0:
            query += self.lora_scale * self.to_q_lora(hidden_states)
            key += self.lora_scale * self.to_k_lora(encoder_hidden_states)
            value += self.lora_scale * self.to_v_lora(encoder_hidden_states)


        ## 全部用spatial temporal
        key, value = self.rearrangeKV_spatial_temporal(key, value, video_length)
        # key, value = self.rearrangeKV_sparse_causal(key, value, video_length)

        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()

        ## store or edit query and key
        assert self.attnStore, "Set self.attnStore first"
        query, key = self.attnStore.self_call(query, key, attn)

        hidden_states = xformers.ops.memory_efficient_attention(
            query, key, value, attn_bias=attention_mask, op=self.attention_op, scale=attn.scale
        )

        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj

        hidden_states = attn.to_out[0](hidden_states, *args)
        if self.lora_scale > 0:
            hidden_states += self.lora_scale * self.to_out_lora(hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def set_attn_processor_with_store(unet, attnStore: AttnStore):

    attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") \
            else 768
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        if cross_attention_dim is None:  # attn1, self attn
            attn_procs[name] = LoRAXFormersSpatialTemporalAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim,
            )
            # attn_procs[name] = XFormersSpatialTemporalAttnProcessor()
        else:  # attn2, cross attn
            attn_procs[name] = LoRAIPAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim,
                ip_scale=0.0, num_tokens=0,
            ).to(attnStore.device, dtype=attnStore.weight_dtype)
            # attn_procs[name] = AttnProcessor()
        attn_procs[name] = attn_procs[name].to(attnStore.device, dtype=attnStore.weight_dtype)
        attn_procs[name].attnStore = attnStore
        attn_procs[name].requires_grad_(False) ## 让LoRA层冻起来

    unet.set_attn_processor(attn_procs)
