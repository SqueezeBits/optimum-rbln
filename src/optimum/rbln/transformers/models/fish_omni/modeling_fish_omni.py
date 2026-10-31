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
import math
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors import safe_open
from torch import nn
from transformers import PreTrainedModel
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextRotaryEmbedding

from ...modeling_outputs import RBLNDecoderOnlyOutput
from ..qwen3_5.modeling_qwen3_5 import _qwen3_5_setup_hybrid_runtime
from ..qwen3_5.qwen3_5_runtime_utils import RBLNQwen3_5RuntimeModel
from ..qwen3_5_moe.modeling_qwen3_5_moe import RBLNQwen3_5MoeForCausalLM
from .fish_omni_architecture import DOWN_SCALE, FP8_BLOCK_SIZE, GATE_UP_SCALE, FishOmni_CausalLMWrapper


# Text config flags fixed by the architecture: (key, default when absent, required value).
_FIXED_TEXT_FLAGS = (
    ("rotary_interleaved", False, False),
    ("use_qk_norm", True, True),
    ("attn_output_gate", True, True),
    ("layernorm_zero_centered_gamma", True, True),
    ("shared_expert_gate", True, True),
    ("norm_topk_prob", True, True),  # FishOmniFp8MoeMLP and the Qwen3.5-MoE block renormalize the top-k weights
)
_TEXT_MODEL_TYPES = ("fish_omni_text", "qwen3_5_moe_text")
_KEY_MAPPING = {r"^model\.language_model\.": "model."}
_LAYER_KEY = re.compile(r"^model\.layers\.(\d+)\.")
_EXPERT_KEY = re.compile(
    r"^(model\.layers\.\d+\.mlp\.experts)\.(\d+)\.(gate|up|down)_proj\.(weight|weight_scale_inv)$"
)


def fish_omni_text_config(config_dict: dict[str, Any], **overrides: Any) -> Qwen3_5MoeTextConfig:
    """Map a FishOmni `text_config` (checkpoint config.json) to a transformers Qwen3.5-MoE text config.

    `use_qk_norm`, `attn_output_gate`, `layernorm_zero_centered_gamma` and `shared_expert_gate` are always on in
    the transformers class, and the MoE blocks always renormalize the top-k routing weights (`norm_topk_prob`), so the
    checkpoint has to ask for exactly that.
    """
    text = dict(config_dict.get("text_config", config_dict))
    if text.get("model_type") not in _TEXT_MODEL_TYPES:
        raise ValueError(f"text_config.model_type={text.get('model_type')!r}; expected one of {_TEXT_MODEL_TYPES}")
    if any(t != "moe" for t in text.get("mlp_layer_types") or ()):
        raise NotImplementedError("every text_config.mlp_layer_types must be moe")
    for key, default, required in _FIXED_TEXT_FLAGS:
        if text.get(key, default) is not required:
            raise NotImplementedError(f"text_config.{key} must be {required}; got {text.get(key)!r}")
    if text.get("tie_word_embeddings", False):
        raise NotImplementedError("FishOmni keeps a separate lm_head")
    rope = text.get("rope_parameters") or {}
    if rope.get("rope_type", "default") != "default":
        raise NotImplementedError("only the default rope_type is supported")
    fields = [
        "vocab_size",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "hidden_act",
        "max_position_embeddings",
        "initializer_range",
        "rms_norm_eps",
        "attention_bias",
        "attention_dropout",
        "head_dim",
        "linear_conv_kernel_dim",
        "linear_key_head_dim",
        "linear_value_head_dim",
        "linear_num_key_heads",
        "linear_num_value_heads",
        "moe_intermediate_size",
        "shared_expert_intermediate_size",
        "num_experts_per_tok",
        "num_experts",
        "layer_types",
        "eos_token_id",
        "pad_token_id",
        "bos_token_id",
        "use_cache",
    ]
    kwargs = {k: text[k] for k in fields if k in text}
    kwargs.update(overrides)
    kwargs["tie_word_embeddings"] = False
    kwargs["rope_parameters"] = {
        "rope_type": "default",
        "rope_theta": rope["rope_theta"],
        "partial_rotary_factor": rope.get("partial_rotary_factor", 0.25),
    }
    return Qwen3_5MoeTextConfig(**kwargs)


def dequantize_block_fp8(
    weight: torch.Tensor, scale_inv: torch.Tensor, block_size: int = FP8_BLOCK_SIZE
) -> torch.Tensor:
    """Dequantize a block-FP8 weight: `weight.float() * scale_inv` with each scale expanded over its block."""
    rows, cols = weight.shape
    scale = scale_inv.float().repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)
    return weight.float() * scale[:rows, :cols]


def _stack_experts_bf16(
    experts: dict[int, dict[str, torch.Tensor]], dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack per-expert tensors into the fused HF layout: gate_up_proj [E, 2I, H], down_proj [E, H, I].

    Each expert is dequantized to fp32 first, then cast to `dtype`.
    """

    def weight(parts: dict[str, torch.Tensor], name: str) -> torch.Tensor:
        w = parts[f"{name}_proj.weight"]
        scale = parts.get(f"{name}_proj.weight_scale_inv")
        return (w.float() if scale is None else dequantize_block_fp8(w, scale)).to(dtype)

    gate_up, down = [], []
    for e in sorted(experts):
        gate_up.append(torch.cat([weight(experts[e], "gate"), weight(experts[e], "up")], dim=0))
        down.append(weight(experts[e], "down"))
    return torch.stack(gate_up), torch.stack(down)


def _stack_experts_fp8(
    experts: dict[int, dict[str, torch.Tensor]], dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack per-expert block-FP8 tensors without dequantizing them.

    Returns gate_up_proj [E, 2I, H] and down_proj [E, H, I] in the checkpoint's float8 dtype, and their
    `weight_scale_inv` blocks [E, 2I/128, H/128] and [E, H/128, I/128] in float32. `dtype` is unused: the scales
    stay at the checkpoint's float32 and the weights at 8 bits.
    """
    gate_up, down, gate_up_scale, down_scale = [], [], [], []
    for e in sorted(experts):
        parts = experts[e]
        for name in ("gate", "up", "down"):
            if f"{name}_proj.weight_scale_inv" not in parts:
                raise ValueError(
                    f"expert {e} {name}_proj has no weight_scale_inv; expert_storage='fp8' needs block FP8"
                )
        gate_up.append(torch.cat([parts["gate_proj.weight"], parts["up_proj.weight"]], dim=0))
        gate_up_scale.append(
            torch.cat([parts["gate_proj.weight_scale_inv"], parts["up_proj.weight_scale_inv"]], dim=0).float()
        )
        down.append(parts["down_proj.weight"])
        down_scale.append(parts["down_proj.weight_scale_inv"].float())
    return torch.stack(gate_up), torch.stack(down), torch.stack(gate_up_scale), torch.stack(down_scale)


def _checkpoint_files(model_dir: Path) -> list[Path]:
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        names = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        return [model_dir / n for n in names]
    return sorted(model_dir.glob("*.safetensors"))


def load_fish_omni_text_state_dict(
    model_dir: Path, config: Qwen3_5MoeTextConfig, dtype: torch.dtype, expert_storage: str = "fp8"
) -> dict[str, torch.Tensor]:
    """Read the text backbone tensors of a FishOmni checkpoint as a Qwen3_5MoeForCausalLM state dict.

    `model.language_model.*` is renamed to `model.*`; `lm_head.weight` is kept; audio decoder and codebook tensors
    are ignored. Per-expert tensors are stacked into the fused expert layout: kept in FP8 with their block scales for
    `expert_storage="fp8"` (`_stack_experts_fp8`, the default), dequantized for `expert_storage="bf16"`
    (`_stack_experts_bf16`).

    Every tensor is a copy: `safe_open(...).get_tensor` returns a view of the file's mmap, and a view kept in the model
    (a tensor already in `dtype`) would keep every shard mapped while the model lives. This takes the place of
    `RBLNMoeLoadMixin`'s `release_checkpoint_mmap_`, which this loader does not go through.
    """
    stackers = {"bf16": _stack_experts_bf16, "fp8": _stack_experts_fp8}
    if expert_storage not in stackers:
        raise ValueError(f'expert_storage must be "fp8" or "bf16", got {expert_storage!r}')
    stack = stackers[expert_storage]
    state: dict[str, torch.Tensor] = {}
    experts: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
    for path in _checkpoint_files(model_dir):
        with safe_open(str(path), framework="pt") as f:
            for key in f.keys():  # noqa: SIM118
                new = key
                for pattern, repl in _KEY_MAPPING.items():
                    new = re.sub(pattern, repl, new)
                if not (new.startswith("model.") or new == "lm_head.weight"):
                    continue
                layer = _LAYER_KEY.match(new)
                if layer and int(layer.group(1)) >= config.num_hidden_layers:
                    continue
                match = _EXPERT_KEY.match(new)
                if match:
                    prefix, idx, proj, kind = match.groups()
                    experts.setdefault(prefix, {}).setdefault(int(idx), {})[f"{proj}_proj.{kind}"] = f.get_tensor(key)
                else:
                    tensor = f.get_tensor(key)
                    state[new] = tensor.to(dtype, copy=True) if tensor.is_floating_point() else tensor.clone()
    for prefix, per_expert in experts.items():
        if len(per_expert) != config.num_experts:
            raise ValueError(f"{prefix}: found {len(per_expert)} experts, config has {config.num_experts}")
        stacked = stack(per_expert, dtype)
        state[f"{prefix}.gate_up_proj"], state[f"{prefix}.down_proj"] = stacked[:2]
        if expert_storage == "fp8":
            state[f"{prefix}.{GATE_UP_SCALE}"], state[f"{prefix}.{DOWN_SCALE}"] = stacked[2:]
    return state


CODEBOOK_EMBEDDINGS_KEY = "audio_decoder.codebook_embeddings.weight"
BACKBONE_EMBEDDING_ARTIFACT = "fish_omni_backbone_embedding"


class FishOmniBackboneEmbedding(nn.Module):
    """FishOmni backbone input rows on the host: `FishOmniAudioDecoder.embed_backbone(frames, backbone.embed(...))`.

    A frame row is `[main token, code 0, code 1, ..., code N-1]` (the prompt rows and the sampled frames of
    fish-speech `generate_drama3`). With `vq_feature_placement: backbone_partial`, a row whose main token is in the
    semantic band takes `(embed_tokens(main) + sum_{k=1..K} codebook_embeddings[k * V + code k]) / sqrt(1 + K)`
    (K = `vq_backbone_codebooks`, V = the audio vocab size); every other row takes `embed_tokens(main)` unchanged.
    Upstream op order and dtypes are kept: `F.embedding_bag(mode="sum")` over the K offset rows, the add, then the
    division by `math.sqrt(1 + K)`.

    `codebook_weight` holds rows [V, (1 + K) * V) of the audio decoder's `codebook_embeddings` table, the only rows the
    backbone input reads; index `(k - 1) * V + code` here is upstream row `k * V + code`.
    """

    def __init__(
        self,
        embed_tokens: nn.Embedding,
        codebook_weight: torch.Tensor,
        audio_vocab_size: int,
        vq_backbone_codebooks: int,
        semantic_start_token_id: int,
        semantic_end_token_id: int,
    ):
        super().__init__()
        if codebook_weight.shape[0] != audio_vocab_size * vq_backbone_codebooks:
            raise ValueError(
                f"codebook_weight has {codebook_weight.shape[0]} rows; expected "
                f"{audio_vocab_size} x {vq_backbone_codebooks} (vq_backbone_codebooks)"
            )
        self.embed_tokens = embed_tokens
        self.register_buffer("codebook_weight", codebook_weight, persistent=False)
        self.audio_vocab_size = audio_vocab_size
        self.vq_backbone_codebooks = vq_backbone_codebooks
        self.semantic_start_token_id = semantic_start_token_id
        self.semantic_end_token_id = semantic_end_token_id

    def is_semantic(self, tokens: torch.Tensor) -> torch.Tensor:
        return (tokens >= self.semantic_start_token_id) & (tokens <= self.semantic_end_token_id)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """frames [..., 1 + num_codebooks] int64 -> inputs_embeds [..., hidden] in the embedding dtype."""
        lead, width = frames.shape[:-1], frames.shape[-1]
        frames = frames.reshape(-1, width)
        text_embeddings = self.embed_tokens(frames[:, 0])
        count = self.vq_backbone_codebooks
        if count == 0:
            extra = text_embeddings.new_zeros(text_embeddings.shape)
        else:
            offsets = torch.arange(0, count, device=frames.device) * self.audio_vocab_size
            rows = frames[:, 2 : 2 + count] + offsets
            extra = F.embedding_bag(rows, self.codebook_weight, mode="sum").to(text_embeddings.device)
        mixed = (text_embeddings + extra) / math.sqrt(1 + count)
        semantic = self.is_semantic(frames[:, 0]).to(text_embeddings.device)
        out = torch.where(semantic[:, None], mixed, text_embeddings)
        return out.reshape(*lead, out.shape[-1])


def fish_omni_backbone_embedding_meta(config_dict: dict[str, Any]) -> dict[str, int]:
    """The backbone-input fields of a FishOmni checkpoint config.json (top level + `audio_decoder_config`)."""
    audio = config_dict.get("audio_decoder_config") or {}
    placement = audio.get("vq_feature_placement", "backbone_partial")
    count = int(audio.get("vq_backbone_codebooks", 0))
    if placement != "backbone_partial" and count != 0:
        raise NotImplementedError(f"vq_feature_placement {placement!r} with vq_backbone_codebooks={count}")
    return {
        "audio_vocab_size": int(audio["vocab_size"]),
        "vq_backbone_codebooks": count,
        "semantic_start_token_id": int(config_dict["semantic_start_token_id"]),
        "semantic_end_token_id": int(config_dict["semantic_end_token_id"]),
    }


def load_fish_omni_backbone_codebooks(model_dir: Path, meta: dict[str, int], dtype: torch.dtype) -> torch.Tensor:
    """Rows [V, (1 + K) * V) of `audio_decoder.codebook_embeddings.weight`, cast to `dtype` like the rest of the
    model (fish-speech loads the audio decoder in the model dtype); a copy, not a view of the file's mmap."""
    vocab, count = meta["audio_vocab_size"], meta["vq_backbone_codebooks"]
    for path in _checkpoint_files(model_dir):
        with safe_open(str(path), framework="pt") as f:
            if CODEBOOK_EMBEDDINGS_KEY in f.keys():  # noqa: SIM118
                table = f.get_slice(CODEBOOK_EMBEDDINGS_KEY)[vocab : (1 + count) * vocab]
                return table.to(dtype, copy=True).contiguous()
    raise ValueError(f"{CODEBOOK_EMBEDDINGS_KEY} not found under {model_dir}")


def assemble_prefill_hidden(
    windows: list[torch.Tensor], query_length: int, attention_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Final-norm hidden of one prefill [1, query_length, H] from the per-window graph outputs [1, chunk, H].

    The windows hold the valid tokens first (right padding in the last one). With an attention mask, the valid rows
    go back to the mask slots and padding rows stay zero, as `RBLNQwen3_5RuntimeModel` places hidden states.
    """
    hidden = torch.cat(windows, dim=1)[:, :query_length]
    if attention_mask is None:
        return hidden
    mask = attention_mask.reshape(-1)
    start = int(torch.nonzero(mask, as_tuple=False)[0].item())
    full = torch.zeros(1, mask.shape[0], hidden.shape[-1], dtype=hidden.dtype)
    full[:, start : start + query_length] = hidden
    return full


class FishOmniRuntimeModel(RBLNQwen3_5RuntimeModel):
    """`RBLNQwen3_5RuntimeModel` for graphs compiled with `output_last_hidden_state`: the last graph output is the
    final-norm hidden; prefill returns `hidden_states=(hidden [1, query_length, H],)` over all windows, decode
    `hidden_states=(hidden [B, 1, H],)`. Logits are read as the parent reads them (output 0)."""

    def _split_outputs(self, out):
        self._final_hidden.append(out[-1])
        return out[0], None

    def prefill_forward(self, inputs, cache_position=None, attention_mask=None, *args, **kwargs):
        self._final_hidden = []
        out = super().prefill_forward(inputs, cache_position, attention_mask, *args, **kwargs)
        query_length = int(attention_mask.sum()) if attention_mask is not None else inputs.shape[1]
        hidden = assemble_prefill_hidden(self._final_hidden, query_length, attention_mask)
        return RBLNDecoderOnlyOutput(
            logits=out.logits, padded_cache_lengths=out.padded_cache_lengths, hidden_states=(hidden,)
        )

    def decode_forward(self, *args, **kwargs):
        self._final_hidden = []
        out = super().decode_forward(*args, **kwargs)
        return RBLNDecoderOnlyOutput(logits=out.logits, hidden_states=(self._final_hidden[-1],))


class RBLNFishOmniForCausalLM(RBLNQwen3_5MoeForCausalLM):
    """
    RBLNFishOmniForCausalLM is the FishOmni text backbone (a Qwen3.5-MoE hybrid decoder with an LM head) optimized for
    RBLN NPUs. It loads the `model.language_model.*` and `lm_head.weight` tensors of a FishOmni checkpoint; the audio
    decoder and codec tensors belong to other classes.

    The checkpoint stores the routed experts per expert in block FP8 (128x128, `weight_scale_inv`). By default
    (`rbln_config.expert_storage="fp8"`) they stay FP8 and run through `custom_moe_glu_group_dequantize`, stored at
    8 bits on the device; with `expert_storage="bf16"` they are dequantized to the rbln dtype while loading and run
    through `custom_moe_glu`. The field is saved in rbln_config.json like every other option.

    This model inherits from [`RBLNDecoderOnlyModelForCausalLM`]. Check the superclass documentation for the generic methods the library implements for all its models.

    **Configuration:**
    This model uses [`RBLNFishOmniForCausalLMConfig`] for configuration. When calling methods like `from_pretrained`
    or `from_model`, the `rbln_config` parameter should be an instance of [`RBLNFishOmniForCausalLMConfig`] or a
    dictionary conforming to its structure. See the [`RBLNFishOmniForCausalLMConfig`] class for all available
    configuration options. The text-to-speech loop (`fish_omni_runtime_utils`) needs `use_inputs_embeds` and
    `output_last_hidden_state`.

    Examples:
        ```python
        from optimum.rbln import RBLNFishOmniForCausalLM

        # Simple usage using rbln_* arguments
        model = RBLNFishOmniForCausalLM.from_pretrained(
            "fishaudio/drama-3",
            export=True,
            rbln_max_seq_len=8192,
            rbln_num_devices=4,
        )

        # Using a config dictionary
        rbln_config = {
            "batch_size": 8,
            "decoder_batch_sizes": [8, 4, 2, 1],
            "max_seq_len": 8192,
            "num_devices": 4,
            "use_inputs_embeds": True,
            "output_last_hidden_state": True,
            "expert_storage": "fp8",
        }
        model = RBLNFishOmniForCausalLM.from_pretrained(
            "fishaudio/drama-3",
            export=True,
            rbln_config=rbln_config,
        )

        # Using a RBLNFishOmniForCausalLMConfig instance (recommended for type checking)
        from optimum.rbln import RBLNFishOmniForCausalLMConfig

        config = RBLNFishOmniForCausalLMConfig(
            batch_size=1,
            max_seq_len=8192,
            num_devices=4,
        )
        model = RBLNFishOmniForCausalLM.from_pretrained(
            "fishaudio/drama-3",
            export=True,
            rbln_config=config,
        )
        ```
    """

    _hf_class = Qwen3_5MoeForCausalLM
    _decoder_wrapper_cls = FishOmni_CausalLMWrapper

    @classmethod
    def _wrap_model_if_needed(cls, model: PreTrainedModel, rbln_config: Any):
        # rbln_config.json records expert_storage, so the experts handed in must have been loaded that way.
        loaded = "fp8" if any(GATE_UP_SCALE in layer.mlp.experts._buffers for layer in model.model.layers) else "bf16"
        if loaded != rbln_config.expert_storage:
            raise ValueError(
                f"the model's experts were loaded with expert_storage={loaded!r} but rbln_config.expert_storage is "
                f"{rbln_config.expert_storage!r}; pass the same rbln_config to get_pytorch_model and from_model."
            )
        return super()._wrap_model_if_needed(model, rbln_config)

    @classmethod
    def get_pytorch_model(
        cls,
        model_id: str | Path,
        revision: str | None = None,
        cache_dir: str | None = None,
        local_files_only: bool = False,
        token: bool | str | None = None,
        rbln_config: Any = None,
        **kwargs: Any,
    ) -> PreTrainedModel:
        if rbln_config is not None and getattr(rbln_config, "quantization", None):
            raise NotImplementedError(
                "FishOmni does not take rbln_config.quantization; the block-FP8 experts are selected with "
                "rbln_config.expert_storage ('fp8' by default, 'bf16' to dequantize while loading)."
            )
        if "expert_storage" in kwargs:
            raise ValueError(
                "expert_storage is an rbln_config field: pass rbln_expert_storage=... or "
                "rbln_config={'expert_storage': ...} so that rbln_config.json records it."
            )
        expert_storage = getattr(rbln_config, "expert_storage", "fp8")
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
        dtype = (
            kwargs.get("dtype") or kwargs.get("torch_dtype") or getattr(rbln_config, "dtype", None) or torch.bfloat16
        )
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)

        # Truncating the stack (for example one layer of each kind in a test) is the only config override.
        overrides = {k: kwargs[k] for k in ("num_hidden_layers", "layer_types") if k in kwargs}
        config_dict = json.loads((model_dir / "config.json").read_text())
        config = fish_omni_text_config(config_dict, **overrides)
        state = load_fish_omni_text_state_dict(model_dir, config, dtype, expert_storage)
        with torch.device("meta"):
            model = Qwen3_5MoeForCausalLM(config)
            if expert_storage == "fp8":
                # Block scales live next to the FP8 experts so load_state_dict checks them like any other tensor.
                for layer in model.model.layers:
                    experts = layer.mlp.experts
                    e, two_i, h = experts.gate_up_proj.shape
                    h_blocks, i_blocks = h // FP8_BLOCK_SIZE, two_i // 2 // FP8_BLOCK_SIZE
                    experts.register_buffer(GATE_UP_SCALE, torch.empty(e, 2 * i_blocks, h_blocks))
                    experts.register_buffer(DOWN_SCALE, torch.empty(e, h_blocks, i_blocks))
        result = model.load_state_dict(state, strict=False, assign=True)
        if result.missing_keys or result.unexpected_keys:
            raise ValueError(
                f"FishOmni load mismatch: missing={result.missing_keys[:5]} unexpected={result.unexpected_keys[:5]}"
            )
        # Non-persistent buffers are not in the checkpoint and are still on the meta device.
        model.model.rotary_emb = Qwen3_5MoeTextRotaryEmbedding(config)
        left = [n for n, t in [*model.named_parameters(), *model.named_buffers()] if t.device.type == "meta"]
        if left:
            raise ValueError(f"FishOmni load left meta tensors: {left[:5]}")
        # The codebook rows of the backbone input (host side, see FishOmniBackboneEmbedding); saved with the
        # artifact when rbln_config.use_inputs_embeds is set.
        if "audio_decoder_config" in config_dict:
            meta = fish_omni_backbone_embedding_meta(config_dict)
            model.fish_omni_backbone_embedding = {
                **meta,
                "codebook_weight": load_fish_omni_backbone_codebooks(model_dir, meta, dtype),
            }
        return model.eval()

    @classmethod
    def save_torch_artifacts(cls, model: PreTrainedModel, save_dir_path: Path, subfolder: str, rbln_config: Any):
        super().save_torch_artifacts(model, save_dir_path, subfolder, rbln_config)
        extra = getattr(model, "fish_omni_backbone_embedding", None)
        if rbln_config.use_inputs_embeds and extra is not None:
            path = save_dir_path / subfolder / "torch_artifacts.pth"
            artifacts = torch.load(path, weights_only=False)
            artifacts[BACKBONE_EMBEDDING_ARTIFACT] = extra
            torch.save(artifacts, path)

    def setup_runtime(self):
        # Same runtimes as Qwen3.5; with output_last_hidden_state they also read the final hidden (the last output).
        runtime_cls = FishOmniRuntimeModel if self.rbln_config.output_last_hidden_state else RBLNQwen3_5RuntimeModel
        _qwen3_5_setup_hybrid_runtime(self, runtime_cls=runtime_cls)

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        self.backbone_embedding = None
        if self.rbln_config.use_inputs_embeds:
            artifacts = torch.load(self.model_save_dir / self.subfolder / "torch_artifacts.pth", weights_only=False)
            extra = artifacts.get(BACKBONE_EMBEDDING_ARTIFACT)
            if extra is not None:
                extra = dict(extra)
                self.backbone_embedding = FishOmniBackboneEmbedding(
                    self.embed_tokens, extra.pop("codebook_weight"), **extra
                )

    def embed_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """FishOmni frame rows [..., 1 + num_codebooks] -> `inputs_embeds` (needs `use_inputs_embeds=True`)."""
        if self.backbone_embedding is None:
            raise ValueError(
                "embed_frames needs an artifact compiled with rbln_config use_inputs_embeds=True from a FishOmni "
                "checkpoint (audio_decoder_config and audio_decoder.codebook_embeddings.weight)."
            )
        with torch.no_grad():
            return self.backbone_embedding(frames)


__all__ = ["RBLNFishOmniForCausalLM"]
