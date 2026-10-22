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

"""The Qwen-Image-2.1 transformer on RBLN NPUs, written against the diffusers implementation.

Original (`diffusers.models.transformers.transformer_qwenimage21`) -> here:
- `QwenImage21Transformer2DModel.forward` with `kv_cache_mode` -> `RBLNQwenImage21Transformer2DModel.forward` over a
  KV cache on the device
- `QwenImage21TransformerBlock` -> called as is by `QwenImage21LayerGroupWrapper.forward` (a group of layers per graph)
- `QwenImage21AttnProcessor` -> `RBLNQwenImage21AttnProcessor` (the paged prefill op over the cache)
- the timestep path (`time_text_embed`, `modulation`, `norm_out`) -> `QwenImage21ModulationWrapper` (one small graph)
- `QwenImage21Rope` -> `_RopeGather` (the same table as cos/sin, gathered in the graph) and `build_segments`
- `QwenImage21KVCache`, `kv_cache_mode="extract"/"cached"` -> `_Request`, `_PrefixRegion`, `_CacheLayout`
- `apply_rotary_emb_qwen` -> `_apply_rope`
The blocks and the other modules (`txt_in`, `img_in`, `norm_out`, `proj_out`) are the diffusers modules themselves.
"""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import rebel
import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_qwenimage21 import (
    QwenImage21Attention,
    QwenImage21AttnProcessor,
    QwenImage21TemporalTimesteps,
    QwenImage21Transformer2DModel,
)
from rebel.compile_context import CompileContext
from torch import nn

from .....configuration_utils import RBLNCompileConfig
from .....modeling import RBLNModel
from .....modeling_base import Preprocessor
from .....utils.logging import get_logger
from ....utils import experimental, graphs_by_name, runtimes_by_name
from ...configurations import RBLNQwenImage21Transformer2DModelConfig


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

    from .....diffusers.modeling_diffusers import RBLNDiffusionMixin, RBLNDiffusionMixinConfig


logger = get_logger(__name__)

MODULATION_GRAPH = "modulation"
# The attention of a layer-group call is told by the length of its `kind_tag` input (the values are not used): causal
# for text, bidirectional for condition images and the target, which share their executors (a condition image's
# output projection is computed and not used).
KIND_TAG = {"img": 1, "txt": 2}
# Rows of the original rotary table (QwenImage21Rope): positions 0..8191, then -1024..-1.
ROPE_TABLE_ROWS = 8192 + 1024


def layer_group_graph_name(start: int, end: int) -> str:
    return f"layers_{start}_{end}"


def rope_index_to_row(index: torch.Tensor) -> torch.Tensor:
    """Maps the original rotary positions (negative ones count from the end of the table) to table rows."""
    return torch.where(index < 0, index + ROPE_TABLE_ROWS, index)


class _RopeGather(nn.Module):
    """The original rotary table as real cos/sin, gathered by `[3, C]` row indices into `[C, head_dim]` (pairwise
    repeated, the layout of `_apply_rope`)."""

    def __init__(self, pos_embed: nn.Module):
        super().__init__()
        self.cos = nn.ModuleList(
            nn.Embedding.from_pretrained(f.real.float().repeat_interleave(2, dim=-1), freeze=True)
            for f in pos_embed.freqs
        )
        self.sin = nn.ModuleList(
            nn.Embedding.from_pretrained(f.imag.float().repeat_interleave(2, dim=-1), freeze=True)
            for f in pos_embed.freqs
        )

    def forward(self, index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cos = torch.cat([table(index[axis]) for axis, table in enumerate(self.cos)], dim=-1)
        sin = torch.cat([table(index[axis]) for axis, table in enumerate(self.sin)], dim=-1)
        return cos, sin


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """The original `apply_rotary_emb_qwen(use_real=False)` with real tables. x `[1, C, H, D]`, cos/sin `[C, D]`."""
    pairs = x.float().unflatten(-1, (-1, 2))
    rotated = torch.stack([-pairs[..., 1], pairs[..., 0]], dim=-1).flatten(-2)
    return (x.float() * cos[None, :, None, :] + rotated * sin[None, :, None, :]).to(x.dtype)


class QwenImage21ModulationWrapper(nn.Module):
    """Timestep embedding → the modulation every layer uses and the output-norm scale, concatenated `[1, 5 * hidden]`.
    One small graph, so that the layer-group graphs take its output instead of each holding the timestep path."""

    def __init__(self, model: QwenImage21Transformer2DModel):
        super().__init__()
        self.timestep_embedder = model.time_text_embed.timestep_embedder
        self.modulation = model.modulation
        self.norm_out = model.norm_out

    def forward(self, timestep_proj: torch.Tensor) -> torch.Tensor:
        """The timestep part of `QwenImage21Transformer2DModel.forward`: `time_text_embed`, `modulation` and the
        `norm_out` scale."""
        temb = self.timestep_embedder(timestep_proj)
        return torch.cat([self.modulation(temb), self.norm_out.linear(self.norm_out.silu(temb))], dim=-1)


class RBLNQwenImage21AttnProcessor(QwenImage21AttnProcessor):
    """`QwenImage21AttnProcessor` for the layer-group graphs: the attention is the paged prefill op over the KV cache
    on the device, in place of the original's attention call per segment of the block-causal mask.

    The blocks pass `rotary_emb`, `attention_mask` and `layer_cache` through to the processor untouched, so the wrapper
    hands over the rotary tables as real cos/sin (the original's are complex), the cache mask as `attention_mask` and
    the layer's cache with the positions written as `layer_cache`: `(kcache, vcache, cache_position, block_tables)`.
    (A tensor kept as state on the processor is not traced as an input by rebel-compiler 0.12.1.dev163.) The chunk is
    one segment of the original's `segments`, `(start, end, is_text)`: a text segment attends causally, an image
    bidirectionally.
    """

    def __init__(self, flash: bool, block_size: int, head_dim: int) -> None:
        super().__init__()
        self.flash, self.block_size = flash, block_size
        self.scale = torch.tensor(head_dim**-0.5)

    def __call__(
        self,
        attn: QwenImage21Attention,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        layer_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        segments: list[tuple[int, int, bool]] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Mirrors `_qwenimage21_prepare_qkv` (the projections, `norm_q`/`norm_k`, the rotary embedding) and the output
        projection of `QwenImage21AttnProcessor.__call__`, with the paged prefill op between them."""
        heads, head_dim, chunk = attn.heads, attn.inner_dim // attn.heads, hidden_states.shape[1]
        query = attn.to_q(hidden_states).unflatten(-1, (heads, head_dim))
        key = attn.to_k(hidden_states).unflatten(-1, (heads, head_dim))
        value = attn.to_v(hidden_states).unflatten(-1, (heads, head_dim))
        query = attn.norm_q(query).to(value.dtype)
        key = attn.norm_k(key).to(value.dtype)
        cos, sin = rotary_emb
        query, key = _apply_rope(query, cos, sin), _apply_rope(key, cos, sin)  # [1, C, H, D]

        kcache, vcache, cache_position, block_tables = layer_cache
        args = {
            "q": query.transpose(1, 2).reshape(1, heads, 1, chunk, head_dim),
            "k": key.transpose(1, 2).unsqueeze(2),
            "v": value.transpose(1, 2).unsqueeze(2),
            "kcache": kcache.unsqueeze(2),
            "vcache": vcache.unsqueeze(2),
            "scale": self.scale,
            "block_table": block_tables,
            "block_size": self.block_size,
            "is_bidirectional": not segments[0][2],
            "mask": attention_mask,
        }
        if self.flash:
            out = torch.ops.rbln_custom_ops.paged_flash_causal_attn_prefill(
                seq=cache_position[:, :1], partition=self.block_size, **args
            )
        else:
            out = torch.ops.rbln_custom_ops.paged_causal_attn_prefill(seq=cache_position[:, :1], **args)
        hidden_states = out.view(1, heads, chunk, head_dim).transpose(1, 2).flatten(2, 3)
        return attn.to_out[1](attn.to_out[0](hidden_states))


class QwenImage21LayerGroupWrapper(nn.Module):
    """Layers `[start, end)` over the device KV cache, for one chunk of text, condition-image or target tokens.

    The joint sequence of the original is `[text | cond image 1 | text | ... | cond image n | text | target]` with
    block-causal attention: causal over text, bidirectional within an image. A chunk writes its keys and values at cache
    positions `cache_position` and attends over the cache (`mask` keeps the valid positions). The kind of the chunk is
    told by the length of `kind_tag`, so one compiled graph holds an executor per kind and bucket, sharing the weights.

    Inputs: x `[C, 4096]` (the first group: text `[C, 4096]` or latents `[C, 64]`), kind_tag, mod `[1, 5 * 4096]`,
    rope_index `[3, C]`, cache_position `[1, C]`, block_tables, mask `[1, capacity]`, then the cache of each layer.
    Output: the hidden state, or for the last group the noise prediction `[C, 64]` of a target chunk (`[1, 64]` for a
    prefix chunk, whose only effect is the cache write).
    """

    def __init__(
        self,
        model: QwenImage21Transformer2DModel,
        start: int,
        end: int,
        rbln_config: RBLNQwenImage21Transformer2DModelConfig,
    ):
        super().__init__()
        self.txt_in, self.img_in = model.txt_in, model.img_in
        self.norm_out, self.proj_out = model.norm_out, model.proj_out
        self.blocks = model.transformer_blocks[start:end]
        self.rope = _RopeGather(model.pos_embed)
        self.first, self.last = start == 0, end == model.config.num_layers
        self.processor = RBLNQwenImage21AttnProcessor(
            rbln_config.attention == "flash", rbln_config.cache_block_size, model.config.attention_head_dim
        )
        for block in self.blocks:
            block.attn.set_processor(self.processor)

    def forward(
        self,
        x: torch.Tensor,
        kind_tag: torch.Tensor,
        mod: torch.Tensor,
        rope_index: torch.Tensor,
        cache_position: torch.Tensor,
        block_tables: torch.Tensor,
        mask: torch.Tensor,
        *caches: torch.Tensor,
    ) -> torch.Tensor:
        """Mirrors `QwenImage21Transformer2DModel.forward` for the layers of the group: the input projection
        `txt_in`/`img_in` in the first group, the blocks as they are (each slices its scales and gates out of
        `modulation`), and `norm_out`/`proj_out` in the last."""
        kind = {tag: kind for kind, tag in KIND_TAG.items()}[kind_tag.shape[0]]
        mod = mod + kind_tag[:1, None] * 0.0  # (kind_tag must be used, or the compiler drops the input)
        if self.first:
            x = self.txt_in(x) if kind == "txt" else self.img_in(x)
        x = x[None]  # [1, C, hidden]
        hidden = x.shape[-1]
        modulation, out_scale = mod[:, : 4 * hidden], mod[:, None, 4 * hidden :]
        rotary_emb = self.rope(rope_index)
        segments = [(0, x.shape[1], kind == "txt")]  # the chunk as one segment of the original's block-causal mask
        for i, block in enumerate(self.blocks):
            layer_cache = (caches[2 * i], caches[2 * i + 1], cache_position, block_tables)
            x = block(
                x, modulation, rotary_emb=rotary_emb, attention_mask=mask, layer_cache=layer_cache, segments=segments
            )
        if not self.last:
            return x[0]
        if kind == "img":
            return self.proj_out(self.norm_out.norm(x) * (1 + out_scale))[0]
        # A text chunk only writes the cache; its output still depends on the layers so that they are kept.
        return x[0, :1, :1].repeat(1, self.proj_out.out_features)


@dataclass
class _Segment:
    kind: str  # txt | img | tgt
    length: int
    rope_rows: torch.Tensor  # [3, length]


def build_segments(image_pad_mask: torch.Tensor, img_shapes: list[tuple[int, int, int]]) -> list[_Segment]:
    """Splits the joint sequence into text, condition-image and target segments, with the rotary table rows the
    original `QwenImage21Rope` gives them. `image_pad_mask` `[S]` marks the image tokens (target included)."""
    is_image = image_pad_mask.tolist()
    segments, cursor, position = [], 0, 0
    for index, (_, height, width) in enumerate(img_shapes):
        block_start = is_image.index(True, cursor)
        text_len = block_start - cursor
        if text_len > 0:
            rows = torch.arange(position, position + text_len)
            segments.append(_Segment("txt", text_len, torch.stack([rows, rows, rows])))
        position += text_len
        frame = torch.full((height * width,), position)
        rows = torch.arange(-(height - height // 2), height // 2).repeat_interleave(width)
        cols = torch.arange(-(width - width // 2), width // 2).repeat(height)
        kind = "tgt" if index == len(img_shapes) - 1 else "img"
        segments.append(_Segment(kind, height * width, rope_index_to_row(torch.stack([frame, rows, cols]))))
        position += max(height, width)
        cursor = block_start + height * width
    if cursor < len(is_image):
        raise NotImplementedError("Text after the target image is not supported (the pipeline never builds it).")
    return segments


@dataclass
class _PrefixRegion:
    """The prefix of one (context, sample) in the cache: written from `start` up to `end`, with the mask of its valid
    positions. A volatile region is rewritten from 0 on every call (the request's prefixes did not all fit)."""

    start: int = 0
    volatile: bool = True
    end: int | None = None
    mask: torch.Tensor | None = None


@dataclass
class _Request:
    """The cache state of a request, the device-side counterpart of the original's `QwenImage21KVCache` over its
    `kv_cache_mode='extract'` and `'cached'` calls: where the next kept prefix region starts, whether the regions
    overflowed (every region is then volatile), and the region of each (context, sample)."""

    cursor: int = 0
    all_volatile: bool = False
    regions: dict[tuple[str, int], _PrefixRegion] = field(default_factory=dict)


class _CacheLayout:
    """The device-side counterpart of the original's `QwenImage21KVCache` layout.

    Cache positions of consecutive chunks. With flash attention a chunk must stay inside one block, so a chunk that
    would cross a block boundary starts at the next block. Raises ValueError when the cache is full.
    """

    def __init__(self, rbln_config: RBLNQwenImage21Transformer2DModelConfig, cursor: int = 0):
        self.flash = rbln_config.attention == "flash"
        self.block_size, self.capacity = rbln_config.cache_block_size, rbln_config.cache_capacity
        self.cursor = cursor

    def place(self, length: int) -> int:
        if self.flash:
            if length > self.block_size:
                raise ValueError(f"A chunk of {length} tokens does not fit in a cache block of {self.block_size}.")
            if self.cursor // self.block_size != (self.cursor + length - 1) // self.block_size:
                self.cursor = (self.cursor // self.block_size + 1) * self.block_size
        start = self.cursor
        self.cursor += length
        if self.cursor > self.capacity:
            raise ValueError(f"The KV cache of {self.capacity} tokens is full.")
        return start


@experimental
class RBLNQwenImage21Transformer2DModel(RBLNModel):
    """
    RBLN implementation of QwenImage21Transformer2DModel for Qwen-Image-2.1 pipelines.

    The text and condition-image keys and values are step-independent (`causal_condition`), so the original prefills
    them at the first step (`kv_cache_mode="extract"`) and later steps compute only the target image
    (`kv_cache_mode="cached"`). The same happens here over a key/value cache that stays on the RBLN devices: the prefix
    is written in chunks (text in pieces of `text_bucket` tokens, each condition image as one chunk) and the target is
    computed after it. The CFG contexts (`cache_context("cond"/"uncond")`) keep their own prefix regions.

    The layers run as a few layer-group graphs (see `layer_groups`), each with one executor per attention kind and bucket;
    the timestep embedding and modulation are one more graph. Only the sinusoidal timestep projection is computed on
    the host, in float32.

    This class inherits from [`RBLNModel`]. Check the superclass documentation for the generic methods
    the library implements for all its models.
    """

    hf_library_name = "diffusers"
    auto_model_class = QwenImage21Transformer2DModel
    _output_class = Transformer2DModelOutput

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        self.time_proj = QwenImage21TemporalTimesteps(timestep_dim=256)
        self.graphs = graphs_by_name(self.model, self.rbln_config)
        self._context = "cond"
        self._request: _Request | None = None
        self._modulations = {}

    @property
    def transformer_blocks(self) -> range:
        # The pipeline only counts the blocks (to size its own cache object, which is not used here).
        return range(self.config.num_layers)

    @contextmanager
    def cache_context(self, name: str) -> Iterator[None]:
        """Marks the CFG context of the calls inside (as the diffusers `CacheMixin`): each keeps its own prefix."""
        previous, self._context = self._context, name
        try:
            yield
        finally:
            self._context = previous

    @classmethod
    def update_rbln_config_using_pipe(
        cls, pipe: "RBLNDiffusionMixin", rbln_config: "RBLNDiffusionMixinConfig", submodule_name: str
    ) -> "RBLNDiffusionMixinConfig":
        # The pipeline config sets the profile it shares with the VAE.
        return rbln_config

    @classmethod
    def get_compiled_model(
        cls, model: "PreTrainedModel", rbln_config: RBLNQwenImage21Transformer2DModelConfig
    ) -> dict[str, rebel.RBLNCompiledModel]:
        devices = rbln_config.device_map
        compiled_models = {}
        for compile_config in rbln_config.compile_cfgs:
            name = compile_config.compiled_model_name
            context = CompileContext(use_weight_sharing=True)
            caches = {}
            if name == MODULATION_GRAPH:
                wrapped = QwenImage21ModulationWrapper(model).eval()
            else:
                start, end = (int(n) for n in name.split("_")[1:])
                wrapped = QwenImage21LayerGroupWrapper(model, start, end, rbln_config).eval()
                # The cache of each layer stays on the device between calls: its example input is the marked tensor.
                for layer in range(start, end):
                    for kind in ("k", "v"):
                        caches[f"{kind}cache_{layer}"] = cls._cache_tensor(rbln_config)
                        context.mark_static_address(caches[f"{kind}cache_{layer}"], f"{kind}cache_{layer}")
            compiled_models[name] = cls.compile(
                wrapped,
                rbln_compile_config=compile_config,
                create_runtimes=rbln_config.create_runtimes,
                device=devices[name],
                compile_context=context,
                example_inputs=cls._example_inputs(compile_config, caches),
            )
        return compiled_models

    @staticmethod
    def _example_inputs(
        compile_config: RBLNCompileConfig, caches: dict[str, torch.Tensor]
    ) -> list[torch.Tensor] | list[list[torch.Tensor]]:
        # The marked caches, valid indices for the integer inputs (rotary rows, cache positions, block tables), zeros
        # otherwise.
        def example(name, shape, dtype):
            if name in caches:
                return caches[name]
            if name in ("cache_position", "block_tables"):
                return torch.arange(shape[-1], dtype=getattr(torch, dtype)).reshape(shape)
            return torch.zeros(shape, dtype=getattr(torch, dtype))

        multiple = compile_config.is_multiple_input_info
        input_infos = compile_config.input_info if multiple else [compile_config.input_info]
        examples = [[example(*info) for info in input_info] for input_info in input_infos]
        return examples if multiple else examples[0]

    @staticmethod
    def _cache_tensor(rbln_config: RBLNQwenImage21Transformer2DModelConfig) -> torch.Tensor:
        return torch.zeros(rbln_config.cache_num_blocks, 32, rbln_config.cache_block_size, 128)

    @classmethod
    def _create_runtimes(
        cls, compiled_models: list[rebel.RBLNCompiledModel], rbln_config: RBLNQwenImage21Transformer2DModelConfig
    ) -> list[rebel.Runtime]:
        return runtimes_by_name(cls, compiled_models, rbln_config)

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Sequence[Preprocessor],
        model: "PreTrainedModel",
        model_config: "PretrainedConfig",
        rbln_config: RBLNQwenImage21Transformer2DModelConfig,
    ) -> RBLNQwenImage21Transformer2DModelConfig:
        num_layers, hidden = (
            model_config.num_layers,
            model_config.num_attention_heads * model_config.attention_head_dim,
        )
        if rbln_config.layer_groups[0][0] != 0 or rbln_config.layer_groups[-1][1] != num_layers:
            raise ValueError(
                f"layer_groups must cover the {num_layers} layers in order, got {rbln_config.layer_groups}"
            )
        cache_shape = list(cls._cache_tensor(rbln_config).shape)
        compile_cfgs = [
            RBLNCompileConfig(
                compiled_model_name=MODULATION_GRAPH, input_info=[("timestep_proj", [1, 256], "float32")]
            )
        ]
        # The executors of a graph share its weights, laid out for the executor compiled first: the largest
        # bidirectional bucket (the target, run on every step) goes first, the text last.
        buckets = sorted(set(rbln_config.image_buckets + rbln_config.target_buckets), reverse=True)
        executors = [("img", n) for n in buckets] + [("txt", rbln_config.text_bucket)]
        for start, end in rbln_config.layer_groups:
            input_info = []
            for kind, chunk in executors:
                width = (hidden if kind == "txt" else model_config.in_channels) if start == 0 else hidden
                info = [
                    ("x", [chunk, width], "float32"),
                    ("kind_tag", [KIND_TAG[kind]], "float32"),
                    ("mod", [1, 5 * hidden], "float32"),
                    ("rope_index", [3, chunk], "int64"),
                    ("cache_position", [1, chunk], "int32"),
                    ("block_tables", [rbln_config.cache_num_blocks], "int16"),
                    ("mask", [1, rbln_config.cache_capacity], "float32"),
                ]
                for layer in range(start, end):
                    info += [(f"{kind_}cache_{layer}", cache_shape, "float32") for kind_ in ("k", "v")]
                input_info.append(info)
            compile_cfgs.append(
                RBLNCompileConfig(compiled_model_name=layer_group_graph_name(start, end), input_info=input_info)
            )
        rbln_config.set_compile_cfgs(compile_cfgs)
        return rbln_config

    def _modulation(self, timestep: float) -> torch.Tensor:
        """The `mod` input of the layer groups for a timestep (the prefix uses 0, like the original)."""
        if timestep not in self._modulations:
            if len(self._modulations) > 64:
                self._modulations.clear()
            proj = self.time_proj(torch.tensor([timestep], dtype=torch.float32)).float()
            self._modulations[timestep] = self.graphs[MODULATION_GRAPH](proj)
        return self._modulations[timestep]

    def _run_chunk(
        self,
        kind: str,
        x: torch.Tensor,
        timestep: float,
        rope: torch.Tensor,
        position: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Runs a chunk through the layer groups; `timestep` is 0 for the prefix, like the original."""
        args = (
            torch.zeros(KIND_TAG["txt" if kind == "txt" else "img"]),
            self._modulation(timestep),
            rope,
            position,
            torch.arange(self.rbln_config.cache_num_blocks, dtype=torch.int16),
            mask,
        )
        for start, end in self.rbln_config.layer_groups:
            x = self.graphs[layer_group_graph_name(start, end)](x, *args)
        return x

    def _bucket(self, kind: str, length: int) -> int:
        buckets = {
            "txt": [self.rbln_config.text_bucket],
            "img": self.rbln_config.image_buckets,
            "tgt": self.rbln_config.target_buckets,
        }[kind]
        for bucket in buckets:
            if bucket >= length:
                return bucket
        option = {"img": "image_buckets", "tgt": "target_buckets"}.get(kind, "text_bucket")
        raise ValueError(
            f"{length} {'condition-image' if kind == 'img' else 'target'} tokens do not fit the compiled buckets "
            f"(`{option}={buckets}`, profile {self.rbln_config.profile!r})."
        )

    def _prefix_chunks(self, segments: list[_Segment]) -> list[tuple[str, int, _Segment, int, int]]:
        """(kind, bucket, segment, offset, length) of the prefix chunks in order: text in pieces of `text_bucket`
        tokens (text is causal, so this gives the same result), every condition image whole."""
        chunks = []
        for segment in segments:
            if segment.kind == "txt":
                for offset in range(0, segment.length, self.rbln_config.text_bucket):
                    length = min(self.rbln_config.text_bucket, segment.length - offset)
                    chunks.append(("txt", self.rbln_config.text_bucket, segment, offset, length))
            elif segment.kind == "img":
                chunks.append(("img", self._bucket("img", segment.length), segment, 0, segment.length))
        return chunks

    def _prefix_end(self, segments: list[_Segment], start: int, target_bucket: int) -> int | None:
        """Where the prefix ends when written from `start`, or None if it and the target do not fit."""
        layout = _CacheLayout(self.rbln_config, start)
        try:
            for _, bucket, *_ in self._prefix_chunks(segments):
                layout.place(bucket)
            end = layout.cursor
            layout.place(target_bucket)
        except ValueError:
            return None
        return end

    def _write_prefix(
        self,
        segments: list[_Segment],
        text: torch.Tensor,
        text_valid: torch.Tensor,
        cond_latents: torch.Tensor,
        start: int,
    ) -> tuple[torch.Tensor, int]:
        """Writes the prefix chunks to the cache from `start`; returns the valid-position mask and the prefix end.

        The original's `kv_cache_mode='extract'`: the text and condition-image keys and values go to the cache, chunk
        by chunk, with `timestep` 0.
        Each chunk sees the mask of the chunks up to itself (later positions may still hold another request's keys).
        """
        layout = _CacheLayout(self.rbln_config, start)
        mask = torch.zeros(1, self.rbln_config.cache_capacity)
        text_ptr = image_ptr = 0
        for kind, bucket, segment, offset, length in self._prefix_chunks(segments):
            if kind == "txt":
                rows = text[text_ptr + offset : text_ptr + offset + length]
                valid = text_valid[text_ptr + offset : text_ptr + offset + length].float()
                if offset + length == segment.length:
                    text_ptr += segment.length
            else:
                rows, valid = cond_latents[image_ptr : image_ptr + length], torch.ones(length)
                image_ptr += length
            position = layout.place(bucket)
            mask[0, position : position + length] = valid
            x, rope = self._pad_chunk(rows, segment.rope_rows[:, offset : offset + length], bucket)
            self._run_chunk(kind, x, 0.0, rope, self._positions(position, bucket), mask.clone())
        return mask, layout.cursor

    @staticmethod
    def _pad_chunk(rows: torch.Tensor, rope_rows: torch.Tensor, bucket: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.zeros(bucket, rows.shape[-1])
        x[: rows.shape[0]] = rows.float()
        rope = torch.zeros(3, bucket, dtype=torch.long)
        rope[:, : rows.shape[0]] = rope_rows
        return x, rope

    @staticmethod
    def _positions(start: int, bucket: int) -> torch.Tensor:
        return (start + torch.arange(bucket, dtype=torch.int32))[None]

    def _forward_one(
        self,
        key: tuple[str, int],
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        text_valid: torch.Tensor,
        slots: torch.Tensor,
        segments: list[_Segment],
        mode: str | None,
    ) -> torch.Tensor:
        """One sample: write (or reuse) its prefix, then compute the target over it.

        The original's `kv_cache_mode='extract'` call followed by its `'cached'` calls.
        """
        target = segments[-1]
        num_target = target.length
        num_cond = hidden_states.shape[0] - num_target
        text = encoder_hidden_states[~slots]  # the image-slot rows are replaced by latents, like in the original
        text_valid = text_valid[~slots]
        target_bucket = self._bucket("tgt", num_target)

        request = self._request
        region = request.regions.get(key)
        if region is None:
            end = None if request.all_volatile else self._prefix_end(segments, request.cursor, target_bucket)
            if end is None:  # out of room: from now on every context rewrites its prefix from 0 on every call
                request.all_volatile = True
                region = _PrefixRegion()
            else:  # a prefix region kept for the whole request
                region = _PrefixRegion(start=request.cursor, volatile=False)
                request.cursor = end
            request.regions[key] = region
        if request.all_volatile:
            region.start, region.volatile = 0, True
        if region.mask is None or mode is None or region.volatile:
            if self._prefix_end(segments, region.start, target_bucket) is None:
                raise ValueError(
                    f"The prompt, condition images and target ({num_target} tokens) do not fit the KV cache of "
                    f"{self.rbln_config.cache_capacity} tokens (profile {self.rbln_config.profile!r})."
                )
            region.mask, region.end = self._write_prefix(
                segments, text, text_valid, hidden_states[:num_cond], region.start
            )

        # The target goes right after the prefixes (attention costs grow with the position); its keys and values are
        # rewritten on every call.
        prefix_end = region.end if request.all_volatile else request.cursor
        position = _CacheLayout(self.rbln_config, prefix_end).place(target_bucket)
        mask = region.mask.clone()
        mask[0, position : position + num_target] = 1
        x, rope = self._pad_chunk(hidden_states[num_cond:], target.rope_rows, target_bucket)
        out = self._run_chunk("tgt", x, float(timestep), rope, self._positions(position, target_bucket), mask)
        return out[:num_target]

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        img_shapes: list[list[tuple[int, int, int]]],
        img_mask: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor | None = None,
        attention_kwargs: dict[str, Any] | None = None,
        kv_cache: Any | None = None,
        kv_cache_mode: str | None = None,
        return_dict: bool = True,
    ) -> Transformer2DModelOutput | tuple:
        """
        Forward pass for the RBLN-optimized QwenImage21Transformer2DModel.

        Args:
            hidden_states (torch.Tensor): Packed latents of shape `(batch_size, image_seq_len, in_channels)`,
                condition images first and the target image last.
            encoder_hidden_states (torch.Tensor): Prompt embeddings of shape `(batch_size, text_seq_len, 4096)`, with
                a row per image slot.
            timestep (torch.Tensor): Timesteps in `[0, 1]` of shape `(batch_size,)`.
            img_shapes (list): `(frames, height, width)` of every condition image and the target, per sample. The
                layout of the first sample is used, like the original.
            img_mask (torch.Tensor): Marks the image slots among the prompt tokens (each expands to 4 tokens).
            encoder_hidden_states_mask (torch.Tensor | None): 1 for the prompt tokens, 0 for batch padding.
            attention_kwargs (dict | None): Unused; accepted for compatibility with the diffusers signature.
            kv_cache (Any | None): The pipeline's cache object, unused: the cache is on the device.
            kv_cache_mode (str | None): "extract" writes the prefix (first step), "cached" reuses it, None rewrites it
                on every call.
            return_dict (bool): Whether or not to return a [`~diffusers.models.modeling_output.Transformer2DModelOutput`] instead of a plain tuple.

        Returns:
            (`~diffusers.models.modeling_output.Transformer2DModelOutput` | tuple)
        """
        repeats = torch.where(img_mask, 4, 1)[0]
        pad_mask = torch.repeat_interleave(img_mask[0], repeats)
        if int(pad_mask.sum()) != hidden_states.shape[1]:
            raise RuntimeError(
                f"shape mismatch: {hidden_states.shape[1]} image tokens for {int(pad_mask.sum())} image slots"
            )
        segments = build_segments(pad_mask, img_shapes[0])
        if self._context == "cond" and kv_cache_mode in (None, "extract"):  # a request starts with its cond call
            self._request = _Request()
        if self._request is None:
            raise RuntimeError(
                "The first call of a request must be in the 'cond' context with kv_cache_mode 'extract'."
            )

        slots = img_mask[0][: encoder_hidden_states.shape[1]]
        timesteps = torch.as_tensor(timestep).reshape(-1)
        outputs = []
        for b in range(hidden_states.shape[0]):
            text_valid = (
                torch.ones(encoder_hidden_states.shape[1], dtype=torch.bool)
                if encoder_hidden_states_mask is None
                else encoder_hidden_states_mask[b].bool()
            )
            outputs.append(
                self._forward_one(
                    (self._context, b),
                    hidden_states[b],
                    timesteps[b if timesteps.numel() > 1 else 0],
                    encoder_hidden_states[b],
                    text_valid,
                    slots,
                    segments,
                    kv_cache_mode,
                )
            )
        return self._prepare_output(torch.stack(outputs).float(), return_dict)


__all__ = [
    "RBLNQwenImage21Transformer2DModel",
]
