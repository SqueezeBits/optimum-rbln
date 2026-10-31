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

from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import rebel
import torch
from huggingface_hub import snapshot_download
from rebel.compile_context import CompileContext
from transformers import PretrainedConfig

from ....configuration_utils import RBLNCompileConfig
from ....modeling import RBLNModel
from .configuration_fish_omni_codec import RBLNFishOmniCodecConfig
from .fish_omni_codec_architecture import (
    FishOmniCodecArgs,
    FishOmniCodecDecoder,
    FishOmniCodecNextWindow,
    FishOmniCodecWindow,
    codec_code_vectors,
    decode_codec_windows,
    load_fish_omni_codec_state_dict,
)


if TYPE_CHECKING:
    from ....modeling_base import Preprocessor

CODEC_CHECKPOINT_NAME = "codec.pth"
HOST_ARTIFACT_NAME = "torch_codec_codebooks.pth"


class RBLNFishOmniCodec(RBLNModel):
    """
    FishOmni codec decoder (modded DAC vocoder) on RBLN NPUs, decoding code streams window by window.

    Compiled graphs (one `CompileContext` with weight sharing):

    - `decoder`: frames 0 .. W - 1. Inputs `code_vectors` [B, num_codebooks, codebook_dim, W] (host codebook
      lookups), `cos` / `sin` [W, head_dim] fp32. Outputs audio [B, 1, W * samples_per_frame], the post_module output
      z [B, W, latent_dim] (time-major) and every post_module layer's K/V [L, 2, B, H, W, D];
    - `decoder_next`: each following window of W frames, with the previous window's last `context_frames` z frames
      and its K/V as inputs; the first `context_frames * samples_per_frame` output samples are dropped on the host.

    `decode(codes)` runs the windows; the result equals upstream `DAC.from_indices` over the whole stream (see
    `fish_omni_codec_architecture`).

    This model inherits from [`RBLNModel`]. Check the superclass documentation for the generic methods the library
    implements for all its models.

    **Configuration:**
    This model uses [`RBLNFishOmniCodecConfig`] for configuration. When calling methods like `from_model` or
    `from_pretrained`, the `rbln_config` parameter should be an instance of [`RBLNFishOmniCodecConfig`] or a
    dictionary conforming to its structure. See the [`RBLNFishOmniCodecConfig`] class for all available configuration
    options.

    Examples:
        ```python
        import torch

        from optimum.rbln import RBLNFishOmniCodec

        # The codec decoder is built from the checkpoint's codec.pth, then compiled from that module
        torch_model = RBLNFishOmniCodec.get_pytorch_model("fishaudio/drama-3", dtype=torch.float32)
        model = RBLNFishOmniCodec.from_model(
            torch_model,
            rbln_config={"batch_size": 1, "window_frames": 128, "context_frames": 16},
            model_save_dir="drama-3-codec",
        )

        # codes [batch, num_codebooks, frames] -> audio [batch, 1, frames * 2048]
        audio = model.decode(torch.zeros(1, 10, 256, dtype=torch.long))
        ```
    """

    _supports_non_fp32 = True

    def __post_init__(self, **kwargs: Any) -> None:
        path = Path(self.model_save_dir) / self.subfolder / HOST_ARTIFACT_NAME
        self.codebooks = torch.load(path, map_location="cpu", weights_only=True) if path.exists() else None

    @classmethod
    def _from_pretrained(
        cls, model_id: str | Path, config: PretrainedConfig | None = None, subfolder: str = "", **kwargs: Any
    ):
        # The saved config.json has no `model_type` (FishOmni has no transformers AutoConfig entry), so it is read as
        # a plain PretrainedConfig instead of through AutoConfig.
        path = Path(model_id) / subfolder / "config.json"
        if config is None and path.is_file():
            config = PretrainedConfig.from_json_file(str(path))
        return super()._from_pretrained(model_id=model_id, config=config, subfolder=subfolder, **kwargs)

    @classmethod
    def get_pytorch_model(
        cls,
        model_id: str | Path,
        revision: str | None = None,
        cache_dir: str | None = None,
        local_files_only: bool = False,
        token: bool | str | None = None,
        rbln_config: RBLNFishOmniCodecConfig | None = None,
        **kwargs: Any,
    ) -> FishOmniCodecDecoder:
        model_dir = Path(model_id)
        if not model_dir.is_dir():
            model_dir = Path(
                snapshot_download(
                    str(model_id),
                    revision=revision,
                    cache_dir=cache_dir,
                    token=token,
                    local_files_only=local_files_only,
                    allow_patterns=[CODEC_CHECKPOINT_NAME],
                )
            )
        dtype = kwargs.get("dtype") or getattr(rbln_config, "dtype", None) or torch.float32
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        model = build_fish_omni_codec(dtype=torch.float32)
        state, codebooks = load_fish_omni_codec_state_dict(model_dir / CODEC_CHECKPOINT_NAME, model.args)
        model.load_state_dict(state, strict=True)
        model.codebooks = codebooks
        return model.to(dtype).eval()

    @classmethod
    def save_torch_artifacts(
        cls, model: FishOmniCodecDecoder, save_dir_path: Path, subfolder: str, rbln_config: RBLNFishOmniCodecConfig
    ) -> None:
        codebooks = getattr(model, "codebooks", None)
        if codebooks:
            torch.save(codebooks, save_dir_path / subfolder / HOST_ARTIFACT_NAME)

    @classmethod
    def _graph_module(cls, model: FishOmniCodecDecoder, name: str, window: int) -> torch.nn.Module:
        graph = FishOmniCodecWindow if name == "decoder" else FishOmniCodecNextWindow
        return graph(model, window).eval()

    @classmethod
    @torch.inference_mode()
    def get_compiled_model(cls, model: FishOmniCodecDecoder, rbln_config: RBLNFishOmniCodecConfig):
        context = CompileContext(use_weight_sharing=True)
        compiled = {}
        for compile_cfg in rbln_config.compile_cfgs:
            name = compile_cfg.compiled_model_name
            compiled[name] = cls.compile(
                cls._graph_module(model, name, rbln_config.window_frames),
                compile_cfg,
                create_runtimes=rbln_config.create_runtimes,
                device=rbln_config.device,
                example_inputs=compile_cfg.get_dummy_inputs(fill=0),
                compile_context=context,
            )
        return compiled

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Sequence["Preprocessor"] | None = None,
        model: FishOmniCodecDecoder | None = None,
        model_config: PretrainedConfig | None = None,
        rbln_config: RBLNFishOmniCodecConfig | None = None,
    ) -> RBLNFishOmniCodecConfig:
        args = model.args
        batch, window, dtype = rbln_config.batch_size, rbln_config.window_frames, rbln_config.dtype
        window_info = [
            ("code_vectors", [batch, args.num_codebooks, args.codebook_dim, window], dtype),
            ("cos", [window, args.post_head_dim], "float32"),
            ("sin", [window, args.post_head_dim], "float32"),
        ]
        compile_cfgs = [RBLNCompileConfig(compiled_model_name="decoder", input_info=window_info)]
        if rbln_config.compile_next:
            kv_shape = [args.post_n_layer, 2, batch, args.post_n_head, window, args.post_head_dim]
            next_info = [
                *window_info,
                ("z_context", [batch, rbln_config.context_frames, args.latent_dim], dtype),
                ("kv_prev", kv_shape, dtype),
            ]
            compile_cfgs.append(RBLNCompileConfig(compiled_model_name="decoder_next", input_info=next_info))
        rbln_config.set_compile_cfgs(compile_cfgs)
        return rbln_config

    @classmethod
    def _create_runtimes(
        cls, compiled_models: list[rebel.RBLNCompiledModel], rbln_config: RBLNFishOmniCodecConfig
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

    def runtime(self, name: str) -> rebel.Runtime:
        names = [cfg.compiled_model_name for cfg in self.rbln_config.compile_cfgs]
        return self.model[names.index(name)]

    @torch.inference_mode()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """codes [B, num_codebooks, frames] -> audio [B, 1, frames * samples_per_frame] (fp32)."""
        if self.codebooks is None:
            raise RuntimeError(f"{HOST_ARTIFACT_NAME} is missing; compile from the checkpoint to save the codebooks")
        args = FishOmniCodecArgs(**{k: getattr(self.config, k) for k in FishOmniCodecArgs.__dataclass_fields__})
        dtype = self.rbln_config.dtype
        dtype = dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
        first, following = self.runtime("decoder"), None
        if codes.shape[-1] > self.rbln_config.window_frames:
            following = self.runtime("decoder_next")

        def run_first(vectors, cos, sin):
            return first(vectors.to(dtype), cos, sin)

        def run_next(vectors, cos, sin, z_context, kv_prev):
            return following(vectors.to(dtype), cos, sin, z_context.to(dtype), kv_prev.to(dtype))

        vectors = codec_code_vectors(codes, self.codebooks, args)
        return decode_codec_windows(
            run_first, run_next, vectors, args, self.rbln_config.window_frames, self.rbln_config.context_frames
        ).float()

    def forward(self, codes: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.decode(codes)


def build_fish_omni_codec(
    args: FishOmniCodecArgs | None = None, dtype: torch.dtype = torch.float32
) -> FishOmniCodecDecoder:
    """Codec decoder device module with its `config` set to a `PretrainedConfig` of the decoder arguments."""
    args = args or FishOmniCodecArgs()
    return FishOmniCodecDecoder(args, PretrainedConfig(**asdict(args))).to(dtype)


__all__ = ["RBLNFishOmniCodec"]
