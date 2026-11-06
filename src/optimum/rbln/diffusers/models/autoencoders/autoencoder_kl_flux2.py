# Copyright 2026 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The FLUX.2 / FLUX.3 VAE on RBLN NPUs.

Nothing of the diffusers `AutoencoderKLFlux2` is re-expressed here: the encoder and decoder graphs are its own
`encoder`, `quant_conv` and `decode` traced through the library's `_VAEEncoder` and `_VAEDecoder`, as for
`RBLNAutoencoderKL`. This subclass adds what the FLUX pipelines need on top of that class: an encoder graph per
reference-image size (`encoder_sample_size`) sharing the encoder weights, `encode` picking the graph of its input's
size, and the batch-norm statistics (`bn`) the pipelines read outside `encode`/`decode`, kept on the host. Tiling and
slicing are not supported, as the graphs are static.
"""

import math
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import rebel
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLFlux2
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.upsampling import Upsample2D
from rebel import RBLNCompileError
from rebel.compile_context import CompileContext

from ....configuration_utils import RBLNCompileConfig, RBLNModelConfig
from ....modeling_base import Preprocessor
from ....utils.logging import get_logger
from ...configurations import RBLNAutoencoderKLFlux2Config
from .autoencoder_kl import RBLNAutoencoderKL
from .vae import (
    RBLNRuntimeVAEEncoder,
    _VAEDecoder,
    _VAEEncoder,
)


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

    from ...modeling_diffusers import RBLNDiffusionMixin, RBLNDiffusionMixinConfig


logger = get_logger(__name__)


def _encoder_graph_name(size: tuple[int, int]) -> str:
    return f"encoder_{size[0]}x{size[1]}"


class _ConvTransposeUpsample2D(torch.nn.Module):
    """`Upsample2D` of the decoder (nearest x2, then its conv) with the nearest x2 as a depthwise transposed conv with a
    2x2 kernel of ones, which gives the same values. A workaround: the compiler fails on the `F.interpolate` resize
    at some sizes (an 832 side, e.g. 1248x832) and when the decoder is split over the chiplets."""

    def __init__(self, upsample: Upsample2D):
        super().__init__()
        self.conv = upsample.conv

    def forward(self, hidden_states: torch.Tensor, output_size: int | None = None) -> torch.Tensor:
        channels = hidden_states.shape[1]
        ones = hidden_states.new_ones(channels, 1, 2, 2)
        return self.conv(F.conv_transpose2d(hidden_states, ones, stride=2, groups=channels))


def _decoder_with_conv_transpose_upsampling(model: AutoencoderKLFlux2) -> _VAEDecoder:
    """`_VAEDecoder` of `model` with its nearest-x2 upsamplers replaced by `_ConvTransposeUpsample2D` (the model is
    only traced afterwards; its `bn` statistics, the one part used at run time, are untouched)."""
    for block in model.decoder.up_blocks:
        for i, up in enumerate(block.upsamplers or []):
            if (
                isinstance(up, Upsample2D)
                and up.interpolate
                and up.use_conv
                and not up.use_conv_transpose
                and up.norm is None
                and up.name == "conv"
            ):
                block.upsamplers[i] = _ConvTransposeUpsample2D(up)
    return _VAEDecoder(model).eval()


class RBLNAutoencoderKLFlux2(RBLNAutoencoderKL):
    """
    RBLN implementation of AutoencoderKLFlux2, the VAE of FLUX.2 and FLUX.3 pipelines.

    The decoder is compiled like [`RBLNAutoencoderKL`]'s for `sample_size`, and the encoder once per size in
    `encoder_sample_size` (FLUX pipelines encode reference images of their own sizes); `encode` runs the graph of its
    input's size. FLUX pipelines normalize the packed latents with the VAE's batch-norm running statistics (`vae.bn`)
    outside `encode`/`decode`, so those statistics are kept on the host and exposed as `bn`.

    This class inherits from [`RBLNAutoencoderKL`]. Check the superclass documentation for the generic methods
    the library implements for all its models.
    """

    auto_model_class = AutoencoderKLFlux2
    _rbln_config_class = RBLNAutoencoderKLFlux2Config

    def __post_init__(self, **kwargs):
        # The parent sets up the decoder (the last graph); the encoder graphs are looked up by size here.
        super().__post_init__(**kwargs)
        graphs = {cfg.compiled_model_name: self.model[i] for i, cfg in enumerate(self.rbln_config.compile_cfgs)}
        self.encoders = {
            size: RBLNRuntimeVAEEncoder(runtime=graphs[_encoder_graph_name(size)], main_input_name="x")
            for size in self.rbln_config.encoder_sample_size or []
        }

        artifacts = torch.load(self.model_save_dir / self.subfolder / "torch_artifacts.pth", weights_only=False)
        self.bn = torch.nn.BatchNorm2d(
            math.prod(self.config.patch_size) * self.config.latent_channels,
            eps=self.config.batch_norm_eps,
            affine=False,
        ).eval()
        self.bn.load_state_dict(artifacts["bn"])

    @classmethod
    def update_rbln_config_using_pipe(
        cls, pipe: "RBLNDiffusionMixin", rbln_config: "RBLNDiffusionMixinConfig", submodule_name: str
    ) -> "RBLNDiffusionMixinConfig":
        # FLUX pipeline configs set the image size; the transformer has no `sample_size` to derive it from.
        if getattr(rbln_config, submodule_name).sample_size is None:
            raise ValueError(
                f"The pipeline config must set `{submodule_name}.sample_size` for RBLNAutoencoderKLFlux2."
            )
        return rbln_config

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Sequence[Preprocessor],
        model: "PreTrainedModel",
        model_config: "PretrainedConfig",
        rbln_config: RBLNAutoencoderKLFlux2Config,
    ) -> RBLNAutoencoderKLFlux2Config:
        # The parent fills in the sizes and the decoder graph; the encoder graphs are one per size here.
        rbln_config = super()._update_rbln_config(preprocessors, model, model_config, rbln_config)
        decoder_cfgs = [cfg for cfg in rbln_config.compile_cfgs if cfg.compiled_model_name == "decoder"]
        if rbln_config.uses_encoder and rbln_config.encoder_sample_size is None:
            rbln_config.encoder_sample_size = [tuple(rbln_config.sample_size)]

        # Largest first (a workaround): under weight sharing the weights are registered by the first graph compiled,
        # and a smaller first graph leaves the mid-block attention weights unregistered for the others.
        encoder_cfgs = [
            RBLNCompileConfig(
                compiled_model_name=_encoder_graph_name(size),
                input_info=[("x", [rbln_config.batch_size, rbln_config.in_channels, *size], rbln_config.dtype)],
            )
            for size in sorted(rbln_config.encoder_sample_size or [], key=lambda s: (-s[0] * s[1], s))
        ]
        rbln_config.set_compile_cfgs(encoder_cfgs + decoder_cfgs)
        return rbln_config

    @classmethod
    @torch.inference_mode()
    def get_compiled_model(
        cls, model, rbln_config: RBLNAutoencoderKLFlux2Config
    ) -> dict[str, rebel.RBLNCompiledModel]:
        # The encoder graphs share the encoder weights through one compile context; the decoder has its own weights.
        # A workaround: a size whose mid-block attention does not lower to the same kernel (a token count that is not
        # a multiple of 64) cannot reuse the shared weights and is compiled on its own.
        context = CompileContext(use_weight_sharing=True)
        compiled_models = {}
        for compile_config in rbln_config.compile_cfgs:
            name = compile_config.compiled_model_name
            compile_kwargs = {
                "rbln_compile_config": compile_config,
                "create_runtimes": rbln_config.create_runtimes,
                "device": rbln_config.device_map[name],
            }
            if name == "decoder":
                compiled_models[name] = cls.compile(_decoder_with_conv_transpose_upsampling(model), **compile_kwargs)
                continue
            try:
                compiled_models[name] = cls.compile(
                    _VAEEncoder(model).eval(), compile_context=context, **compile_kwargs
                )
            except RBLNCompileError:
                logger.warning(
                    f"The compiler cannot share the encoder weights with the {name} graph; compiling it alone."
                )
                compiled_models[name] = cls.compile(_VAEEncoder(model).eval(), **compile_kwargs)
        return compiled_models

    @classmethod
    def _create_runtimes(
        cls, compiled_models: list[rebel.RBLNCompiledModel], rbln_config: RBLNAutoencoderKLFlux2Config
    ) -> list[rebel.Runtime]:
        names = [cfg.compiled_model_name for cfg in rbln_config.compile_cfgs]
        if any(name not in rbln_config.device_map for name in names):
            cls._raise_missing_compiled_file_error(names)
        return [
            rebel.Runtime(
                compiled_model,
                tensor_type="pt",
                device=rbln_config.device_map[name],
                activate_profiler=rbln_config.activate_profiler,
                timeout=rbln_config.timeout,
            )
            for name, compiled_model in zip(names, compiled_models, strict=True)
        ]

    def encode(
        self, x: torch.FloatTensor, return_dict: bool = True, **kwargs: dict[str, Any]
    ) -> torch.FloatTensor | AutoencoderKLOutput:
        """
        Encode an input image into a latent representation, with the encoder graph compiled for its size.

        Args:
            x: The input image to encode, of shape `(batch_size, 3, height, width)`.
            return_dict:
                Whether to return output as a dictionary. Defaults to True.
            kwargs: Additional arguments to pass to the encoder.

        Returns:
            The latent representation or AutoencoderKLOutput if return_dict=True
        """
        size = tuple(x.shape[-2:])
        if size not in self.encoders:
            raise ValueError(
                f"No encoder was compiled for {size[0]}x{size[1]} images; compiled sizes: "
                f"{sorted(self.encoders)} (`encoder_sample_size`)."
            )
        posterior = self.encoders[size].encode(x.to(self.rbln_config.dtype))
        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    @classmethod
    def save_torch_artifacts(
        cls,
        model: "PreTrainedModel",
        save_dir_path: Path,
        subfolder: str,
        rbln_config: RBLNModelConfig,
    ):
        torch.save({"bn": model.bn.state_dict()}, save_dir_path / subfolder / "torch_artifacts.pth")


__all__ = [
    "RBLNAutoencoderKLFlux2",
]
