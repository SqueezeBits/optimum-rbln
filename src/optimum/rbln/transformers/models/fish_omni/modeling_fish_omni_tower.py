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

import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import rebel
import torch
from huggingface_hub import snapshot_download
from rebel.compile_context import CompileContext
from safetensors import safe_open
from transformers import PretrainedConfig

from ....configuration_utils import RBLNCompileConfig
from ....modeling import RBLNModel
from ....utils.logging import get_logger
from .configuration_fish_omni_tower import RBLNFishOmniTowerConfig
from .fish_omni_tower_architecture import (
    TOWER_DEVICE_PREFIXES,
    TOWER_HOST_EMBEDDING_KEYS,
    FishOmniTowerArgs,
    FishOmniTowerDecodeFirst,
    FishOmniTowerDecodeResidual,
    FishOmniTowerModel,
    FishOmniTowerPrefill,
    permute_tower_rope_rows,
)


if TYPE_CHECKING:
    from ....modeling_base import Preprocessor

logger = get_logger(__name__)

HOST_ARTIFACT_NAME = "torch_tower_host_embeddings.pth"


def _kv_names(layer_index: int) -> tuple[str, str]:
    return f"past_key_values_{2 * layer_index}", f"past_key_values_{2 * layer_index + 1}"


class RBLNFishOmniTower(RBLNModel):
    """
    FishOmni codebook tower (codebook-MTP audio decoder) on RBLN NPUs.

    Compiled graphs (one `CompileContext`, one static KV cache per layer shared by all graphs):

    - `prefill`: teacher-forced pass of every layer over a chunk of prompt rows (batch 1, writes all caches);
    - `decode_1`: shared layers + residual layer 1 + head 1;
    - `decode_k`, k = 2 .. num_codebooks - 1: residual layer k + head k.

    The host samples a code between consecutive decode graphs and looks up code embeddings, so every graph takes
    already-embedded codes (`layer_embeds`). Multi-graph structure follows `RBLNWhisperForConditionalGeneration`
    (one `RBLNCompileConfig` per graph, static tensors marked once in a shared `CompileContext`).

    This model inherits from [`RBLNModel`]. Check the superclass documentation for the generic methods the library
    implements for all its models.

    **Configuration:**
    This model uses [`RBLNFishOmniTowerConfig`] for configuration. When calling methods like `from_model` or
    `from_pretrained`, the `rbln_config` parameter should be an instance of [`RBLNFishOmniTowerConfig`] or a
    dictionary conforming to its structure. See the [`RBLNFishOmniTowerConfig`] class for all available configuration
    options.

    Examples:
        ```python
        import torch

        from optimum.rbln import RBLNFishOmniTower

        # The tower is built from the checkpoint's audio decoder tensors (FishOmni has no transformers class),
        # then compiled from that module
        torch_model = RBLNFishOmniTower.get_pytorch_model("fishaudio/drama-3", dtype=torch.float32)
        model = RBLNFishOmniTower.from_model(
            torch_model,
            rbln_config={"batch_size": 8, "max_seq_len": 8192},
            model_save_dir="drama-3-tower",
        )

        # Load the compiled tower
        model = RBLNFishOmniTower.from_pretrained("drama-3-tower", export=False)
        ```
    """

    _supports_non_fp32 = True

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
        rbln_config: RBLNFishOmniTowerConfig | None = None,
        **kwargs: Any,
    ) -> FishOmniTowerModel:
        model_dir = Path(model_id)
        if not model_dir.is_dir():
            model_dir = Path(
                snapshot_download(
                    str(model_id),
                    revision=revision,
                    cache_dir=cache_dir,
                    token=token,
                    local_files_only=local_files_only,
                    allow_patterns=["*.json", "*.safetensors"],
                )
            )
        dtype = kwargs.get("dtype") or getattr(rbln_config, "dtype", None) or torch.bfloat16
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        config_dict = json.loads((model_dir / "config.json").read_text())["audio_decoder_config"]
        model = build_fish_omni_tower(config_dict, dtype=dtype)
        state, host = load_fish_omni_tower_state_dict(model_dir, model.args)
        model.load_state_dict({k: v.to(dtype) for k, v in state.items()}, strict=True)
        # `get_tensor` returns a view of the file's mmap and `.to(dtype)` keeps it when no cast happens (a bf16
        # checkpoint read at the default bf16); load_state_dict copies the device weights, the host tables are kept.
        model.host_embeddings = {k: v.to(dtype, copy=True) for k, v in host.items()}
        return model.eval()

    @classmethod
    def save_torch_artifacts(
        cls, model: FishOmniTowerModel, save_dir_path: Path, subfolder: str, rbln_config: RBLNFishOmniTowerConfig
    ) -> None:
        host = getattr(model, "host_embeddings", None)
        if host:
            torch.save(host, save_dir_path / subfolder / HOST_ARTIFACT_NAME)

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Sequence["Preprocessor"] | None = None,
        model: FishOmniTowerModel | None = None,
        model_config: PretrainedConfig | None = None,
        rbln_config: RBLNFishOmniTowerConfig | None = None,
    ) -> RBLNFishOmniTowerConfig:
        args = FishOmniTowerArgs.from_config(model_config)
        dtype = rbln_config.dtype
        batch, chunk = rbln_config.batch_size, rbln_config.prefill_chunk_size
        kv_shape = [batch, args.n_local_heads, rbln_config.max_seq_len, args.head_dim]

        def kv_info(layers: Sequence[int]) -> list[tuple[str, list[int], Any]]:
            return [(name, kv_shape, dtype) for i in layers for name in _kv_names(i)]

        def step_info(rows: int, length: int, n_embeds: int | None) -> list[tuple[str, list[int], Any]]:
            embed_shape = [rows, length, args.dim] if n_embeds is None else [n_embeds, rows, length, args.dim]
            return [
                ("hidden", [rows, length, args.dim], dtype),
                ("layer_embeds", embed_shape, dtype),
                ("cos", [rows, length, args.rotary_dim], "float32"),
                ("sin", [rows, length, args.rotary_dim], "float32"),
                ("cache_position", [rows, 1], "int32"),
                ("block_tables", [rows, 1], "int16"),
            ]

        compile_cfgs = []
        if rbln_config.compile_prefill:
            prefill_info = step_info(1, chunk, args.num_layers)
            prefill_info[5] = ("block_tables", [1], "int16")
            compile_cfgs.append(
                RBLNCompileConfig(
                    compiled_model_name="prefill", input_info=prefill_info + kv_info(range(args.num_layers))
                )
            )
        shared = args.shared_enrichment_layers
        decode_graphs = rbln_config.decode_graphs or list(range(1, args.num_residual_layers + 1))
        rbln_config.decode_graphs = decode_graphs
        for k in decode_graphs:
            if k == 1:
                info = step_info(batch, 1, shared + 1) + kv_info(range(shared + 1))
            else:
                info = step_info(batch, 1, None) + kv_info([shared + k - 1])
            compile_cfgs.append(RBLNCompileConfig(compiled_model_name=f"decode_{k}", input_info=info))
        rbln_config.set_compile_cfgs(compile_cfgs)
        return rbln_config

    @classmethod
    def _graph_module(cls, model: FishOmniTowerModel, name: str) -> torch.nn.Module:
        if name == "prefill":
            return FishOmniTowerPrefill(model).eval()
        k = int(name.split("_")[1])
        return (FishOmniTowerDecodeFirst(model) if k == 1 else FishOmniTowerDecodeResidual(model, k)).eval()

    @classmethod
    @torch.inference_mode()
    def get_compiled_model(cls, model: FishOmniTowerModel, rbln_config: RBLNFishOmniTowerConfig):
        context = CompileContext(use_weight_sharing=True)
        static_tensors: dict[str, torch.Tensor] = {}
        compiled = {}
        for compile_cfg in rbln_config.compile_cfgs:
            kv_names = [name for name, _, _ in compile_cfg.input_info if name.startswith("past_key_values")]
            new_kv = [name for name in kv_names if name not in static_tensors]
            example_inputs = compile_cfg.get_dummy_inputs(
                fill=0, static_tensors=static_tensors, meta_tensor_names=new_kv
            )
            for (name, _, _), tensor in zip(compile_cfg.input_info, example_inputs, strict=False):
                if name in new_kv:
                    static_tensors[name] = tensor
                    context.mark_static_address(tensor, name)
            logger.info(f"compiling FishOmni tower graph {compile_cfg.compiled_model_name}")
            original_linear = torch.nn.functional.linear
            torch.nn.functional.linear = torch.ops.rbln_custom_ops.linear
            try:
                compiled[compile_cfg.compiled_model_name] = cls.compile(
                    cls._graph_module(model, compile_cfg.compiled_model_name),
                    compile_cfg,
                    create_runtimes=rbln_config.create_runtimes,
                    device=rbln_config.device,
                    example_inputs=example_inputs,
                    compile_context=context,
                )
            finally:
                torch.nn.functional.linear = original_linear
        return compiled

    @classmethod
    def _create_runtimes(
        cls, compiled_models: list[rebel.RBLNCompiledModel], rbln_config: RBLNFishOmniTowerConfig
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
        """The runtime of one compiled graph (`prefill`, `decode_1`, ...); the host loop drives them in order."""
        names = [cfg.compiled_model_name for cfg in self.rbln_config.compile_cfgs]
        return self.model[names.index(name)]

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "RBLNFishOmniTower is driven graph by graph (runtime('prefill'), runtime('decode_k')) with host sampling"
        )


def build_fish_omni_tower(config_dict: dict[str, Any], dtype: torch.dtype = torch.bfloat16) -> FishOmniTowerModel:
    """Tower device module with its `config` set to a `PretrainedConfig` of `audio_decoder_config`."""
    args = FishOmniTowerArgs.from_config(config_dict)
    config = PretrainedConfig(**{k: v for k, v in config_dict.items() if k != "model_type"})
    model = FishOmniTowerModel(args, config).to(dtype)
    return model


def load_fish_omni_tower_state_dict(
    model_dir: Path, args: FishOmniTowerArgs
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Device state dict (RoPE rows permuted) and host embedding tables from the checkpoint's safetensors."""
    index_path = model_dir / "model.safetensors.index.json"
    files = (
        sorted(set(json.loads(index_path.read_text())["weight_map"].values()))
        if index_path.exists()
        else [p.name for p in model_dir.glob("*.safetensors")]
    )
    state, host = {}, {}
    for file in files:
        with safe_open(model_dir / file, framework="pt") as f:
            for key in f.keys():  # noqa: SIM118  safe_open handles are not iterable
                if key.startswith("audio_decoder.codebook_embeddings."):
                    host["codebook_embeddings.weight"] = f.get_tensor(key)
                    continue
                for old, new in TOWER_DEVICE_PREFIXES.items():
                    if key.startswith(old):
                        name = new + key[len(old) :]
                        if name.endswith(TOWER_HOST_EMBEDDING_KEYS):
                            host[name] = f.get_tensor(key)
                        else:
                            state[name] = f.get_tensor(key)
                        break
    return permute_tower_rope_rows(state, args), host


__all__ = ["RBLNFishOmniTower"]
