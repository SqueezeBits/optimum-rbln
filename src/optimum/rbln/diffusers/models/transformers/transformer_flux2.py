# Copyright 2025 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_flux2 import (
    Flux2Attention,
    Flux2AttnProcessor,
    Flux2ParallelSelfAttention,
    Flux2ParallelSelfAttnProcessor,
    Flux2PosEmbed,
    Flux2Transformer2DModel,
    _get_qkv_projections,
    dispatch_attention_fn,
)
from transformers import PretrainedConfig

from ....configuration_utils import RBLNCompileConfig, RBLNModelConfig
from ....modeling import RBLNModel
from ...configurations import RBLNFlux2Transformer2DModelConfig


if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer, PreTrainedModel

    from ...modeling_diffusers import RBLNDiffusionMixin, RBLNDiffusionMixinConfig


class Flux2Transformer2DModelWrapper(torch.nn.Module):
    def __init__(self, model: "Flux2Transformer2DModel") -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ):
        return self.model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
            return_dict=False,
        )


# aten::rms_norm is not supported.
class RBLNFlux2RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float, elementwise_affine: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_parameter("weight", None)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        if self.weight is not None:
            hidden_states = hidden_states * self.weight
        return hidden_states


# aten::outer used in Flux2PosEmbed is not supported.
class RBLNFlux2PosEmbed(nn.Module):
    def __init__(self, theta: int, axes_dim: list[int]):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    @staticmethod
    def _get_1d_rotary_pos_embed(dim: int, pos: torch.Tensor, theta: float, freqs_dtype: torch.dtype):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device) / dim))
        freqs = pos.unsqueeze(-1) * freqs.unsqueeze(0)
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()
        return freqs_cos, freqs_sin

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        cos_out = []
        sin_out = []
        pos = ids.float()
        # RBLN graph conversion fails on this rotary path when the intermediate frequency tensor is float64.
        freqs_dtype = torch.float32

        for i in range(len(self.axes_dim)):
            cos, sin = self._get_1d_rotary_pos_embed(
                self.axes_dim[i],
                pos[..., i],
                theta=self.theta,
                freqs_dtype=freqs_dtype,
            )
            cos_out.append(cos)
            sin_out.append(sin)

        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)
        return freqs_cos, freqs_sin


def rbln_apply_rotary_emb(x: torch.Tensor, freqs_cis: Tuple[torch.Tensor, torch.Tensor], sequence_dim: int = 1):
    cos, sin = freqs_cis
    if sequence_dim == 2:
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
    elif sequence_dim == 1:
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
    else:
        raise ValueError(f"`sequence_dim={sequence_dim}` but should be 1 or 2.")

    cos, sin = cos.to(x.device), sin.to(x.device)
    rotary_impl = os.getenv("RBLN_FLUX2_DEBUG_ROTARY_IMPL", "interleaved_even_odd")

    if rotary_impl == "interleaved_even_odd":
        cos_half = cos[..., ::2]
        sin_half = sin[..., ::2]
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        out_even = x_even * cos_half - x_odd * sin_half
        out_odd = x_odd * cos_half + x_even * sin_half
        return torch.stack([out_even, out_odd], dim=-1).flatten(3).to(x.dtype)

    if rotary_impl == "contiguous_half":
        x_first = x[..., : x.shape[-1] // 2]
        x_second = x[..., x.shape[-1] // 2 :]
        x_rotated = torch.cat([-x_second, x_first], dim=-1)
        return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

    raise ValueError(
        "RBLN_FLUX2_DEBUG_ROTARY_IMPL must be one of "
        "'interleaved_even_odd' or 'contiguous_half', "
        f"got {rotary_impl!r}"
    )


def _get_rotary_tensor_layout() -> str:
    layout = os.getenv("RBLN_FLUX2_DEBUG_ROTARY_LAYOUT", "bshd")
    if layout not in {"bshd", "bhsd"}:
        raise ValueError(
            "RBLN_FLUX2_DEBUG_ROTARY_LAYOUT must be one of 'bshd' or 'bhsd', "
            f"got {layout!r}"
        )
    return layout


class RBLNFlux2AttnProcessor(Flux2AttnProcessor):
    def __call__(
        self,
        attn: "Flux2Attention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        rotary_layout = _get_rotary_tensor_layout()
        if rotary_layout == "bhsd":
            query = query.permute(0, 2, 1, 3)
            key = key.permute(0, 2, 1, 3)
            value = value.permute(0, 2, 1, 3)

            if image_rotary_emb is not None:
                query = rbln_apply_rotary_emb(query, image_rotary_emb, sequence_dim=2)
                key = rbln_apply_rotary_emb(key, image_rotary_emb, sequence_dim=2)

            hidden_states = dispatch_attention_fn(
                query,
                key,
                value,
                attn_mask=attention_mask,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )
            hidden_states = hidden_states.permute(0, 2, 1, 3)
        else:
            if image_rotary_emb is not None:
                query = rbln_apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
                key = rbln_apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

            hidden_states = dispatch_attention_fn(
                query,
                key,
                value,
                attn_mask=attention_mask,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class RBLNFlux2ParallelSelfAttnProcessor(Flux2ParallelSelfAttnProcessor):
    def __call__(
        self,
        attn: "Flux2ParallelSelfAttention",
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = attn.to_qkv_mlp_proj(hidden_states)
        qkv, mlp_hidden_states = torch.split(
            hidden_states, [3 * attn.inner_dim, attn.mlp_hidden_dim * attn.mlp_mult_factor], dim=-1
        )

        query, key, value = qkv.chunk(3, dim=-1)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        rotary_layout = _get_rotary_tensor_layout()
        if rotary_layout == "bhsd":
            query = query.permute(0, 2, 1, 3)
            key = key.permute(0, 2, 1, 3)
            value = value.permute(0, 2, 1, 3)

            if image_rotary_emb is not None:
                query = rbln_apply_rotary_emb(query, image_rotary_emb, sequence_dim=2)
                key = rbln_apply_rotary_emb(key, image_rotary_emb, sequence_dim=2)

            hidden_states = dispatch_attention_fn(
                query,
                key,
                value,
                attn_mask=attention_mask,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )
            hidden_states = hidden_states.permute(0, 2, 1, 3)
        else:
            if image_rotary_emb is not None:
                query = rbln_apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
                key = rbln_apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

            hidden_states = dispatch_attention_fn(
                query,
                key,
                value,
                attn_mask=attention_mask,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        mlp_hidden_states = attn.mlp_act_fn(mlp_hidden_states)
        hidden_states = torch.cat([hidden_states, mlp_hidden_states], dim=-1)
        hidden_states = attn.to_out(hidden_states)

        return hidden_states


class RBLNFlux2Transformer2DModel(RBLNModel):
    hf_library_name = "diffusers"
    auto_model_class = Flux2Transformer2DModel
    _output_class = Transformer2DModelOutput

    @classmethod
    def _wrap_model_if_needed(cls, model: torch.nn.Module, rbln_config: RBLNModelConfig) -> torch.nn.Module:
        return Flux2Transformer2DModelWrapper(model).eval()

    @classmethod
    def _reconstruct_model_if_needed(cls, model: "PreTrainedModel"):
        cls._replace_unsupported_modules(model)
        return model

    @classmethod
    def _replace_unsupported_modules(cls, module: nn.Module):
        for name, child in list(module.named_children()):
            if isinstance(child, Flux2PosEmbed):
                setattr(module, name, RBLNFlux2PosEmbed(theta=child.theta, axes_dim=child.axes_dim))
                continue

            if isinstance(child, Flux2Attention):
                child.set_processor(RBLNFlux2AttnProcessor())

            if isinstance(child, Flux2ParallelSelfAttention):
                child.set_processor(RBLNFlux2ParallelSelfAttnProcessor())

            if isinstance(child, nn.RMSNorm):
                replacement = RBLNFlux2RMSNorm(
                    hidden_size=child.normalized_shape[0],
                    eps=child.eps,
                    elementwise_affine=child.elementwise_affine,
                )
                if child.weight is not None:
                    replacement.weight.data.copy_(child.weight.data)
                setattr(module, name, replacement)
                continue

            cls._replace_unsupported_modules(child)

    @classmethod
    def update_rbln_config_using_pipe(
        cls, pipe: "RBLNDiffusionMixin", rbln_config: "RBLNDiffusionMixinConfig", submodule_name: str
    ) -> "RBLNDiffusionMixinConfig":
        if rbln_config.transformer.image_size is None:
            if rbln_config.image_size is not None:
                rbln_config.transformer.image_size = rbln_config.image_size
            else:
                default_image_size = pipe.default_sample_size * pipe.vae_scale_factor
                rbln_config.transformer.image_size = (default_image_size, default_image_size)

        if rbln_config.transformer.max_seq_len is None:
            rbln_config.transformer.max_seq_len = rbln_config.max_seq_len

        return rbln_config

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"],
        model: "PreTrainedModel",
        model_config: "PretrainedConfig",
        rbln_config: RBLNFlux2Transformer2DModelConfig,
    ) -> RBLNFlux2Transformer2DModelConfig:
        if rbln_config.image_size is None:
            image_size = (1024, 1024)
        else:
            image_size = rbln_config.image_size

        image_token_length = (image_size[0] // 16) * (image_size[1] // 16)

        input_info = [
            (
                "hidden_states",
                [rbln_config.batch_size, image_token_length, model_config.in_channels],
                "float32",
            ),
            (
                "encoder_hidden_states",
                [rbln_config.batch_size, rbln_config.max_seq_len, model_config.joint_attention_dim],
                "float32",
            ),
            ("timestep", [rbln_config.batch_size], "float32"),
            ("img_ids", [rbln_config.batch_size, image_token_length, 4], "int64"),
            ("txt_ids", [rbln_config.batch_size, rbln_config.max_seq_len, 4], "int64"),
        ]

        if getattr(model_config, "guidance_embeds", False):
            input_info.append(("guidance", [rbln_config.batch_size], "float32"))

        compile_config = RBLNCompileConfig(input_info=input_info)
        rbln_config.set_compile_cfgs([compile_config])
        return rbln_config

    @property
    def compiled_batch_size(self):
        return self.rbln_config.compile_cfgs[0].input_info[0][1][0]

    @contextmanager
    def cache_context(self, name: str):
        yield

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        **kwargs,
    ) -> Union[Transformer2DModelOutput, Tuple]:
        sample_batch_size = hidden_states.size(0)
        if sample_batch_size != self.compiled_batch_size:
            raise ValueError(
                f"Mismatch between transformer's runtime batch size ({sample_batch_size}) and "
                f"compiled batch size ({self.compiled_batch_size}). Adjust the transformer batch size during compilation."
            )

        hidden_states = hidden_states.contiguous()
        if encoder_hidden_states is not None:
            encoder_hidden_states = encoder_hidden_states.contiguous()

        forward_args = [hidden_states, encoder_hidden_states, timestep, img_ids, txt_ids]
        if len(self.rbln_config.compile_cfgs[0].input_info) == 6:
            forward_args.append(guidance)

        return super().forward(*forward_args, return_dict=return_dict)
