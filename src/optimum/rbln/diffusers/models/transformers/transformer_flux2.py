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

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_flux2 import (
    Flux2PosEmbed,
    Flux2Transformer2DModel,
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
        image_rotary_emb_0: torch.Tensor = None,
        image_rotary_emb_1: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ):
        num_txt_tokens = encoder_hidden_states.shape[1]
        timestep = timestep.to(hidden_states.dtype) * 1000

        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = self.model.time_guidance_embed(timestep, guidance)
        double_stream_mod_img = self.model.double_stream_modulation_img(temb)
        double_stream_mod_txt = self.model.double_stream_modulation_txt(temb)
        single_stream_mod = self.model.single_stream_modulation(temb)

        hidden_states = self.model.x_embedder(hidden_states)
        encoder_hidden_states = self.model.context_embedder(encoder_hidden_states)

        concat_rotary_emb = (image_rotary_emb_0, image_rotary_emb_1)

        for block in self.model.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb_mod_img=double_stream_mod_img,
                temb_mod_txt=double_stream_mod_txt,
                image_rotary_emb=concat_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        for block in self.model.single_transformer_blocks:
            hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=None,
                temb_mod=single_stream_mod,
                image_rotary_emb=concat_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        hidden_states = hidden_states[:, num_txt_tokens:, ...]
        hidden_states = self.model.norm_out(hidden_states, temb)
        hidden_states = self.model.proj_out(hidden_states)
        return (hidden_states,)


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


class RBLNFlux2Transformer2DModel(RBLNModel):
    hf_library_name = "diffusers"
    auto_model_class = Flux2Transformer2DModel
    _output_class = Transformer2DModelOutput

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        self.pos_embed = Flux2PosEmbed(theta=self.config.rope_theta, axes_dim=self.config.axes_dims_rope)

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
        rotary_dim = sum(model_config.axes_dims_rope)

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
            ("image_rotary_emb_0", [rbln_config.max_seq_len + image_token_length, rotary_dim], "float32"),
            ("image_rotary_emb_1", [rbln_config.max_seq_len + image_token_length, rotary_dim], "float32"),
        ]

        if getattr(model_config, "guidance_embeds", False):
            input_info.append(("guidance", [rbln_config.batch_size], "float32"))

        compile_config = RBLNCompileConfig(input_info=input_info)
        rbln_config.set_compile_cfgs([compile_config])
        return rbln_config

    @property
    def compiled_batch_size(self):
        return self.rbln_config.compile_cfgs[0].input_info[0][1][0]

    def compute_embedding(self, img_ids: torch.Tensor, txt_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]

        image_rotary_emb = self.pos_embed(img_ids)
        text_rotary_emb = self.pos_embed(txt_ids)
        return (
            torch.cat([text_rotary_emb[0], image_rotary_emb[0]], dim=0).float(),
            torch.cat([text_rotary_emb[1], image_rotary_emb[1]], dim=0).float(),
        )

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

        image_rotary_emb_0, image_rotary_emb_1 = self.compute_embedding(img_ids, txt_ids)
        image_rotary_emb_0 = image_rotary_emb_0.contiguous()
        image_rotary_emb_1 = image_rotary_emb_1.contiguous()

        forward_args = [hidden_states, encoder_hidden_states, timestep, image_rotary_emb_0, image_rotary_emb_1]
        if len(self.rbln_config.compile_cfgs[0].input_info) == 6:
            forward_args.append(guidance)

        return super().forward(*forward_args, return_dict=return_dict)
