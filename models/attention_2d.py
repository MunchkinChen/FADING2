# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention.py

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models import ModelMixin
from diffusers.models import Transformer2DModel
from diffusers.utils import BaseOutput
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention import Attention, FeedForward, AdaLayerNorm

from einops import rearrange, repeat


@dataclass
class Transformer2DModelOutput(BaseOutput):
    sample: torch.FloatTensor


if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


class Transformer2DModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        sample_size: Optional[int] = None,
        num_vector_embeds: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        use_sc_attn: bool = False,
        use_st_attn: bool = False,
    ):
        super().__init__()
        self.use_linear_projection = use_linear_projection
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        # 1. Transformer2DModel can process both standard continous images of shape `(batch_size, num_channels, width, height)` as well as quantized image embeddings of shape `(batch_size, num_image_vectors)`
        # Define whether input is continuous or discrete depending on configuration
        self.is_input_continuous = in_channels is not None
        self.is_input_vectorized = num_vector_embeds is not None

        if self.is_input_continuous and self.is_input_vectorized:
            raise ValueError(
                f"Cannot define both `in_channels`: {in_channels} and `num_vector_embeds`: {num_vector_embeds}. Make"
                " sure that either `in_channels` or `num_vector_embeds` is None."
            )
        elif not self.is_input_continuous and not self.is_input_vectorized:
            raise ValueError(
                f"Has to define either `in_channels`: {in_channels} or `num_vector_embeds`: {num_vector_embeds}. Make"
                " sure that either `in_channels` or `num_vector_embeds` is not None."
            )

        # 2. Define input layers
        if self.is_input_continuous:
            self.in_channels = in_channels

            self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
            if use_linear_projection:
                self.proj_in = nn.Linear(in_channels, inner_dim)
            else:
                self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        else:
            raise NotImplementedError

        # Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                    use_sc_attn=use_sc_attn, ##
                    use_st_attn=True if (d == 0 and use_st_attn) else False, ##
                )
                for d in range(num_layers)
            ]
        )

        # 4. Define output layers
        if use_linear_projection:
            self.proj_out = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, hidden_states, encoder_hidden_states=None, timestep=None, return_dict: bool = True, normal_infer: bool = False):
        # Input
        assert hidden_states.dim() == 5, f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
        video_length = hidden_states.shape[2]
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
        encoder_hidden_states = repeat(encoder_hidden_states, 'b n c -> (b f) n c', f=video_length)

        batch, channel, height, weight = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        if not self.use_linear_projection:
            hidden_states = self.proj_in(hidden_states)
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
        else:
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
            hidden_states = self.proj_in(hidden_states)

        # Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                video_length=video_length,
                normal_infer=normal_infer,
            )

        # Output
        if not self.use_linear_projection:
            hidden_states = (
                hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()
            )
            hidden_states = self.proj_out(hidden_states)
        else:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = (
                hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()
            )

        output = hidden_states + residual

        output = rearrange(output, "(b f) c h w -> b c f h w", f=video_length)
        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        use_sc_attn: bool = False,
        use_st_attn: bool = False,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention
        self.use_ada_layer_norm = num_embeds_ada_norm is not None

        # Attn with temporal modeling  ##
        self.use_sc_attn = use_sc_attn
        self.use_st_attn = use_st_attn

        attn_type = SparseCausalAttention if self.use_sc_attn else Attention  ##
        attn_type = SpatialTemporalAttention if self.use_st_attn else attn_type
        self.attn1 = attn_type(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
        )  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)

        # Cross-Attn
        if cross_attention_dim is not None:
            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )  # is self-attn if encoder_hidden_states is none
        else:
            self.attn2 = None

        self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)

        if cross_attention_dim is not None:
            self.norm2 = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)
        else:
            self.norm2 = None

        # 3. Feed-forward
        self.norm3 = nn.LayerNorm(dim)

    # def set_use_memory_efficient_attention_xformers(self, use_memory_efficient_attention_xformers: bool):
    #     if not is_xformers_available():
    #         print("Here is how to install it")
    #         raise ModuleNotFoundError(
    #             "Refer to https://github.com/facebookresearch/xformers for more information on how to install"
    #             " xformers",
    #             name="xformers",
    #         )
    #     elif not torch.cuda.is_available():
    #         raise ValueError(
    #             "torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is only"
    #             " available for GPU "
    #         )
    #     else:
    #         try:
    #             # Make sure we can run the memory efficient attention
    #             _ = xformers.ops.memory_efficient_attention(
    #                 torch.randn((1, 2, 40), device="cuda"),
    #                 torch.randn((1, 2, 40), device="cuda"),
    #                 torch.randn((1, 2, 40), device="cuda"),
    #             )
    #         except Exception as e:
    #             raise e
    #         self.attn1._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
    #         if self.attn2 is not None:
    #             self.attn2._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
    #         # self.attn_temp._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers

    def forward(self, hidden_states, encoder_hidden_states=None, timestep=None, attention_mask=None,
                video_length=None, normal_infer=False):
        # SparseCausal-Attention
        norm_hidden_states = (
            self.norm1(hidden_states, timestep) if self.use_ada_layer_norm else self.norm1(hidden_states)
        )
        ## 换一下self和cross的位置会出事吗？会。

        if self.only_cross_attention:
            # hidden_states = (
            #     self.attn1(norm_hidden_states, encoder_hidden_states, attention_mask=attention_mask) + hidden_states
            # )
            raise NotImplementedError
        else:
            if self.use_sc_attn or self.use_st_attn:
                hidden_states = self.attn1(
                    norm_hidden_states, attention_mask=attention_mask, video_length=video_length, # normal_infer=normal_infer,
                ) + hidden_states
            else:
                # shape of hidden_states: (b*f, len, dim)
                # hidden_states = self.attn1(norm_hidden_states, attention_mask=attention_mask) + hidden_states
                raise NotImplementedError

        if self.attn2 is not None:
            # Cross-Attention
            norm_hidden_states = (
                self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
            )
            hidden_states = (
                self.attn2(
                    norm_hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask
                )
                + hidden_states
            )
        else:
            raise NotImplementedError



        # Feed-forward
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

        return hidden_states


class SparseCausalAttention(Attention):
    # def forward_sc_attn(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None):
    #     batch_size, sequence_length, _ = hidden_states.shape
    #
    #     encoder_hidden_states = encoder_hidden_states
    #
    #     if self.group_norm is not None:
    #         hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
    #
    #     query = self.to_q(hidden_states)
    #     dim = query.shape[-1]
    #     query = self.head_to_batch_dim(query)
    #
    #     if self.added_kv_proj_dim is not None:
    #         raise NotImplementedError
    #
    #     encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
    #     key = self.to_k(encoder_hidden_states)
    #     value = self.to_v(encoder_hidden_states)
    #
    #     ## sparse causal attention !!
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
    #     ##
    #
    #     key = self.head_to_batch_dim(key)
    #     value = self.head_to_batch_dim(value)
    #
    #     if attention_mask is not None:
    #         if attention_mask.shape[-1] != query.shape[1]:
    #             target_length = query.shape[1]
    #             attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
    #             attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)
    #
    #     # attention, what we cannot get enough of
    #     if self._use_memory_efficient_attention_xformers:
    #         hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
    #         # Some versions of xformers return output in fp32, cast it back to the dtype of the input
    #         hidden_states = hidden_states.to(query.dtype)
    #     else:
    #         if self._slice_size is None or query.shape[0] // self._slice_size == 1:
    #             hidden_states = self._attention(query, key, value, attention_mask)
    #         else:
    #             hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)
    #
    #     # linear proj
    #     hidden_states = self.to_out[0](hidden_states)
    #
    #     # dropout
    #     hidden_states = self.to_out[1](hidden_states)
    #     return hidden_states

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
        # if normal_infer:
        #     return super().forward(
        #         hidden_states=hidden_states,
        #         encoder_hidden_states=encoder_hidden_states,
        #         attention_mask=attention_mask,
        #         # video_length=video_length,
        #     )
        # else:
        #     return self.forward_sc_attn(
        #         hidden_states=hidden_states,
        #         encoder_hidden_states=encoder_hidden_states,
        #         attention_mask=attention_mask,
        #         video_length=video_length,
        #     )
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )

class SpatialTemporalAttention(Attention):
    # def forward_dense_attn(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None):
    #     batch_size, sequence_length, _ = hidden_states.shape
    #
    #     # encoder_hidden_states = encoder_hidden_states
    #
    #     if self.group_norm is not None:
    #         hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
    #
    #     if self.added_kv_proj_dim is not None:
    #         raise NotImplementedError
    #
    #     query = self.to_q(hidden_states)
    #     dim = query.shape[-1]
    #
    #     encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
    #     key = self.to_k(encoder_hidden_states)
    #     value = self.to_v(encoder_hidden_states)
    #
    #     ## dense spatial temporal attention !!
    #     key = rearrange(key, "(b f) n d -> b f n d", f=video_length)
    #     key = key.unsqueeze(1).repeat(1, video_length, 1, 1, 1)  # (b f f n d)
    #     key = rearrange(key, "b f g n d -> (b f) (g n) d")
    #
    #     value = rearrange(value, "(b f) n d -> b f n d", f=video_length)
    #     value = value.unsqueeze(1).repeat(1, video_length, 1, 1, 1)  # (b f f n d)
    #     value = rearrange(value, "b f g n d -> (b f) (g n) d")
    #     ##
    #
    #     query = self.head_to_batch_dim(query)
    #     key = self.head_to_batch_dim(key)
    #     value = self.head_to_batch_dim(value)
    #
    #     if attention_mask is not None:
    #         if attention_mask.shape[-1] != query.shape[1]:
    #             target_length = query.shape[1]
    #             attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
    #             attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)
    #
    #     # attention, what we cannot get enough of
    #     if self._use_memory_efficient_attention_xformers:
    #         hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
    #         # Some versions of xformers return output in fp32, cast it back to the dtype of the input
    #         hidden_states = hidden_states.to(query.dtype)
    #     else:
    #         if self._slice_size is None or query.shape[0] // self._slice_size == 1:
    #             hidden_states = self._attention(query, key, value, attention_mask)
    #         else:
    #             hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)
    #
    #     # linear proj
    #     hidden_states = self.to_out[0](hidden_states)
    #
    #     # dropout
    #     hidden_states = self.to_out[1](hidden_states)
    #     return hidden_states
    
    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs,):
        # if normal_infer:
        #     return super().forward(
        #         hidden_states=hidden_states,
        #         encoder_hidden_states=encoder_hidden_states,
        #         attention_mask=attention_mask,
        #         # video_length=video_length,
        #     )
        # else:
        #     return self.forward_dense_attn(
        #         hidden_states=hidden_states,
        #         encoder_hidden_states=encoder_hidden_states,
        #         attention_mask=attention_mask,
        #         video_length=video_length,
        #     )
        #
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
