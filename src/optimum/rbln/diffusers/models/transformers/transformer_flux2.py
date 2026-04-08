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
from diffusers.models.transformers.transformer_flux2 import Flux2PosEmbed, Flux2Transformer2DModel
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
        is_mps = ids.device.type == "mps"
        is_npu = ids.device.type == "npu"
        freqs_dtype = torch.float32 if (is_mps or is_npu) else torch.float64

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

        forward_args = [hidden_states, encoder_hidden_states, timestep, img_ids, txt_ids]
        if len(self.rbln_config.compile_cfgs[0].input_info) == 6:
            forward_args.append(guidance)

        return super().forward(*forward_args, return_dict=return_dict)
