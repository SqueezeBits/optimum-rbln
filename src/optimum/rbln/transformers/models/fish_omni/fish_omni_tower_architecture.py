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

"""FishOmni codebook tower (codebook-MTP audio decoder) as RBLN-traceable graphs.

The tower predicts codes 1..N-1 of every frame. It has `shared_enrichment_layers` shared layers and `num_codebooks - 1`
residual layers, each a pre-norm transformer block attending over time with its own KV cache, plus one head per
residual layer. Host code samples between the residual graphs, so decoding one frame runs:

    decode_1: shared layers + residual layer 1 + head 1
    decode_k: residual layer k + head k   (k = 2 .. num_codebooks - 1)

and prompt rows are written into all caches by a teacher-forced `prefill` graph over every layer.

Deltas against fish-speech `fish_omni/tower.py` (numbers unchanged, structure changed):
- interleaved partial RoPE becomes rotate-half: Q/K rows of `wqkv` and the `q_norm`/`k_norm` weights are permuted at
  load time (`permute_tower_rope_rows`), so q.k is unchanged and no size-2 trailing-dim select is traced;
- cos/sin are computed on the host in fp32 exactly as upstream and passed in (no gather of a table in the graph);
- attention runs through the RBLN paged causal attention ops over static per-layer KV caches (the upstream causal mask
  `key_pos <= query_pos` is what the causal op applies), instead of `scaled_dot_product_attention`;
- a batch dim is added (upstream is batch 1);
- code embeddings (`semantic_embedding`, `codebook_embedding`, `codebook_embeddings`) stay on the host: each layer takes
  `where(semantic, embedding(codes), 0) (+ unseen codes for shared layer 0)` as its `layer_embeds` input.
"""

import math
from dataclasses import dataclass, fields
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class FishOmniTowerArgs:
    dim: int
    intermediate_size: int
    n_head: int
    n_local_heads: int
    head_dim: int
    norm_eps: float
    rope_base: float
    rotary_percent: float
    vocab_size: int
    num_codebooks: int
    shared_enrichment_layers: int
    vq_backbone_codebooks: int

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.rotary_percent)

    @property
    def num_residual_layers(self) -> int:
        return self.num_codebooks - 1

    @property
    def num_layers(self) -> int:
        return self.shared_enrichment_layers + self.num_residual_layers

    @classmethod
    def from_config(cls, config: Any) -> "FishOmniTowerArgs":
        data = config if isinstance(config, dict) else config.to_dict()
        data = data.get("audio_decoder_config", data)
        if not data.get("rotary_interleaved", True) or not data.get("attention_output_gate", True):
            raise NotImplementedError("the tower graphs assume rotary_interleaved and attention_output_gate")
        if not data.get("layernorm_zero_centered_gamma", True):
            raise NotImplementedError("the tower graphs assume zero-centered norm weights")
        return cls(**{f.name: data[f.name] for f in fields(cls)})


def rope_row_permutation(head_dim: int, rotary_dim: int) -> Tensor:
    """Index map from interleaved (even, odd) rotary pairs to rotate-half order; dims past rotary_dim stay."""
    return torch.cat(
        (
            torch.arange(0, rotary_dim, 2),
            torch.arange(1, rotary_dim, 2),
            torch.arange(rotary_dim, head_dim),
        )
    )


def tower_rope_cos_sin(positions: Tensor, args: FishOmniTowerArgs) -> tuple[Tensor, Tensor]:
    """fp32 cos/sin [*positions.shape, rotary_dim] in rotate-half layout, computed like upstream `rotate`."""
    width = args.rotary_dim
    exponents = torch.arange(0, width, 2, dtype=torch.float32)
    angles = positions.float()[..., None] * args.rope_base ** (-exponents / width)
    cos, sin = angles.cos(), angles.sin()
    return torch.cat((cos, cos), -1), torch.cat((sin, sin), -1)


class ZeroCenteredRMSNorm(nn.Module):
    """RMSNorm scaling by 1 + weight, computed in fp32 (as upstream fish_omni/layers.py)."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        out = x.float()
        out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * (1.0 + self.weight.float())).type_as(x)


class FP32OutputLinear(nn.Linear):
    """Upstream keeps the FP16 down projection output in fp32; other dtypes run a plain linear."""

    def forward(self, x: Tensor) -> Tensor:
        if x.dtype != torch.float16:
            return super().forward(x)
        out = F.linear(x.float(), self.weight.float())
        return out if self.bias is None else out + self.bias.float()


def rotate_half_partial(x: Tensor, cos: Tensor, sin: Tensor, rotary_dim: int) -> Tensor:
    """Rotate-half partial RoPE in fp32 on rows permuted by `rope_row_permutation`.

    Upstream computes (even * cos - odd * sin, odd * cos + even * sin) per interleaved pair; after the permute the
    evens are the first half and the odds the second half, so the same products land in permuted order.
    """
    half = rotary_dim // 2
    rot = x[..., :rotary_dim].float()
    first, second = rot[..., :half], rot[..., half:]
    c, s = cos[..., :half], sin[..., :half]
    rotated = torch.cat((first * c - second * s, second * c + first * s), dim=-1)
    return torch.cat((rotated.to(x.dtype), x[..., rotary_dim:]), dim=-1)


def eager_paged_causal_attention(
    q: Tensor, k: Tensor, v: Tensor, kcache: Tensor, vcache: Tensor, seq: Tensor, scale: Tensor, block_table: Tensor
) -> Tensor:
    """CPU reference of the paged causal attention ops with one block per sequence (block_size = cache length).

    q [B, KVH, G, T, D], k / v [B, KVH, 1, T, D], caches [blocks, KVH, 1, S, D], seq [B, 1] start positions,
    block_table [B, 1] (decode) or [1] (prefill). Writes the T new rows at seq..seq+T-1 of each row's block, then
    attends every query at position p to cache rows 0..p (upstream mask `key_pos <= query_pos`).
    The rebel CPU kernels of these ops are placeholders, so CPU equivalence checks run through this function.
    """
    batch, _kv_heads, _groups, length, _head_dim = q.shape
    out = torch.empty_like(q)
    blocks = block_table.reshape(-1)
    for b in range(batch):
        block, start = int(blocks[b]), int(seq.reshape(-1)[b])
        kcache[block, :, 0, start : start + length] = k[b, :, 0]
        vcache[block, :, 0, start : start + length] = v[b, :, 0]
        keys, values = kcache[block, :, 0, : start + length], vcache[block, :, 0, : start + length]
        query_pos = torch.arange(start, start + length)[:, None]
        mask = torch.arange(start + length)[None, :] <= query_pos
        scores = torch.einsum("hgtd,hsd->hgts", q[b], keys) * scale
        probs = scores.masked_fill(~mask, float("-inf")).softmax(-1)
        out[b] = torch.einsum("hgts,hsd->hgtd", probs, values)
    return out


class TowerAttention(nn.Module):
    """Gated attention over time; `wqkv` rows are [Q | Gate | K | V] with Q/K rows already RoPE-permuted."""

    # CPU checks only: run `eager_paged_causal_attention` instead of the rbln custom ops.
    eager_attention = False

    def __init__(self, args: FishOmniTowerArgs) -> None:
        super().__init__()
        self.args = args
        q_width = args.n_head * args.head_dim
        kv_width = args.n_local_heads * args.head_dim
        self.wqkv = nn.Linear(args.dim, 2 * q_width + 2 * kv_width, bias=False)
        self.wo = nn.Linear(q_width, args.dim, bias=False)
        self.q_norm = ZeroCenteredRMSNorm(args.head_dim, args.norm_eps)
        self.k_norm = ZeroCenteredRMSNorm(args.head_dim, args.norm_eps)
        self.scale = 1.0 / math.sqrt(args.head_dim)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        k_cache: Tensor,
        v_cache: Tensor,
        seq_position: Tensor,
        block_tables: Tensor,
        phase: str,
    ) -> Tensor:
        args = self.args
        batch, length, _ = x.shape
        q_width, kv_width = args.n_head * args.head_dim, args.n_local_heads * args.head_dim
        q, gate, k, v = self.wqkv(x).split((q_width, q_width, kv_width, kv_width), -1)
        q = self.q_norm(q.reshape(batch, length, args.n_head, args.head_dim))
        k = self.k_norm(k.reshape(batch, length, args.n_local_heads, args.head_dim))
        cos, sin = cos.unsqueeze(2), sin.unsqueeze(2)
        q = rotate_half_partial(q, cos, sin, args.rotary_dim).transpose(1, 2)
        k = rotate_half_partial(k, cos, sin, args.rotary_dim).transpose(1, 2)
        v = v.reshape(batch, length, args.n_local_heads, args.head_dim).transpose(1, 2)

        groups = args.n_head // args.n_local_heads
        op_args = {
            "q": q.reshape(batch, args.n_local_heads, groups, length, args.head_dim),
            "k": k.unsqueeze(2),
            "v": v.unsqueeze(2),
            "kcache": k_cache.unsqueeze(2),
            "vcache": v_cache.unsqueeze(2),
            "seq": seq_position,
            "scale": torch.tensor(self.scale, dtype=torch.float32),
            "block_table": block_tables,
            "block_size": k_cache.shape[-2],
        }
        if self.eager_attention:
            out = eager_paged_causal_attention(
                *(op_args[n] for n in ("q", "k", "v", "kcache", "vcache", "seq", "scale", "block_table"))
            )
        elif phase == "decode":
            out = torch.ops.rbln_custom_ops.paged_causal_attn_decode(**op_args)
        else:
            op_args["is_bidirectional"] = False
            out = torch.ops.rbln_custom_ops.paged_causal_attn_prefill(**op_args)
        out = out.reshape(batch, args.n_head, length, args.head_dim).transpose(1, 2)
        out = out.reshape(batch, length, q_width)
        return self.wo(out * gate.sigmoid())


class TowerMLP(nn.Module):
    def __init__(self, args: FishOmniTowerArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(args.dim, args.intermediate_size, bias=False)
        self.w3 = nn.Linear(args.dim, args.intermediate_size, bias=False)
        self.w2 = FP32OutputLinear(args.intermediate_size, args.dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TowerTransformerLayer(nn.Module):
    def __init__(self, args: FishOmniTowerArgs) -> None:
        super().__init__()
        self.attention = TowerAttention(args)
        self.attention_norm = ZeroCenteredRMSNorm(args.dim, args.norm_eps)
        self.ffn_norm = ZeroCenteredRMSNorm(args.dim, args.norm_eps)
        self.feed_forward = TowerMLP(args)

    def forward(self, x: Tensor, *attn_args: Any) -> Tensor:
        dtype = self.attention.wqkv.weight.dtype
        residual = x.float() if dtype == torch.float16 else x
        residual = residual + self.attention(self.attention_norm(residual).to(dtype), *attn_args)
        return residual + self.feed_forward(self.ffn_norm(residual).to(dtype))


class TowerLayer(nn.Module):
    """Mix the (host-embedded) code into the hidden state, then one pre-norm block."""

    def __init__(self, args: FishOmniTowerArgs) -> None:
        super().__init__()
        self.enorm = ZeroCenteredRMSNorm(args.dim, args.norm_eps)
        self.hnorm = ZeroCenteredRMSNorm(args.dim, args.norm_eps)
        self.eh_proj = nn.Linear(2 * args.dim, args.dim, bias=False)
        self.transformer_layer = TowerTransformerLayer(args)
        self.final_layernorm = ZeroCenteredRMSNorm(args.dim, args.norm_eps)

    def forward(self, hidden: Tensor, embed: Tensor, *attn_args: Any) -> Tensor:
        mixed = self.eh_proj(torch.cat((self.enorm(embed), self.hnorm(hidden)), dim=-1))
        out = self.final_layernorm(self.transformer_layer(mixed, *attn_args))
        return out.to(mixed.dtype)


class FishOmniTowerModel(nn.Module):
    """Device part of the tower: shared layers, residual layers and codebook heads (embeddings live on the host)."""

    def __init__(self, args: FishOmniTowerArgs, config: Any = None) -> None:
        super().__init__()
        self.args = args
        self.config = config
        self.shared_layers = nn.ModuleList(TowerLayer(args) for _ in range(args.shared_enrichment_layers))
        self.residual_layers = nn.ModuleList(TowerLayer(args) for _ in range(args.num_residual_layers))
        self.codebook_heads = nn.ModuleList(
            nn.Linear(args.dim, args.vocab_size, bias=False) for _ in range(args.num_residual_layers)
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.codebook_heads[0].weight.dtype

    @property
    def layers(self) -> list[TowerLayer]:
        return [*self.shared_layers, *self.residual_layers]


# Checkpoint prefixes of the device part (codebook / semantic embeddings are host tensors).
TOWER_DEVICE_PREFIXES = {
    "audio_decoder_mtp_shared.layers.": "shared_layers.",
    "audio_decoder_mtp.layers.": "residual_layers.",
    "audio_decoder_mtp.heads.": "codebook_heads.",
}
TOWER_HOST_EMBEDDING_KEYS = ("semantic_embedding.weight", "codebook_embedding.weight")


@torch.no_grad()
def permute_tower_rope_rows(state: dict[str, Tensor], args: FishOmniTowerArgs) -> dict[str, Tensor]:
    """Permute Q and K rows of every `wqkv` and the q_norm / k_norm weights into rotate-half order (in place)."""
    perm = rope_row_permutation(args.head_dim, args.rotary_dim)
    q_width, kv_width = args.n_head * args.head_dim, args.n_local_heads * args.head_dim
    for name in list(state):
        if name.endswith("attention.wqkv.weight"):
            w = state[name]
            q, gate, k, v = w.split((q_width, q_width, kv_width, kv_width), 0)
            q = q.reshape(args.n_head, args.head_dim, -1)[:, perm].reshape(q_width, -1)
            k = k.reshape(args.n_local_heads, args.head_dim, -1)[:, perm].reshape(kv_width, -1)
            state[name] = torch.cat((q, gate, k, v), 0).contiguous()
        elif name.endswith(("attention.q_norm.weight", "attention.k_norm.weight")):
            state[name] = state[name][perm].contiguous()
    return state


class _TowerGraph(nn.Module):
    def __init__(self, model: FishOmniTowerModel, phase: str) -> None:
        super().__init__()
        self.model = model
        self.phase = phase


class FishOmniTowerDecodeFirst(_TowerGraph):
    """decode_1: shared layers + residual layer 1 + head 1 for one frame per batch row.

    Inputs: hidden [B, 1, dim], layer_embeds [S + 1, B, 1, dim], cos / sin [B, 1, rotary_dim] fp32,
    cache_position [B, 1] int32, block_tables [B, 1] int16, then K, V caches of layers 0..S (S shared + residual 1).
    Returns (logits [B, 1, vocab], hidden [B, 1, dim]).
    """

    def __init__(self, model: FishOmniTowerModel) -> None:
        super().__init__(model, "decode")
        self.layers = nn.ModuleList([*model.shared_layers, model.residual_layers[0]])
        self.head = model.codebook_heads[0]

    def forward(self, hidden, layer_embeds, cos, sin, cache_position, block_tables, *kv_caches):
        for index, layer in enumerate(self.layers):
            hidden = layer(
                hidden,
                layer_embeds[index],
                cos,
                sin,
                kv_caches[2 * index],
                kv_caches[2 * index + 1],
                cache_position,
                block_tables,
                self.phase,
            )
        return self.head(hidden), hidden


class FishOmniTowerDecodeResidual(_TowerGraph):
    """decode_k (k >= 2): residual layer k + head k. Inputs as decode_1 with embed [B, 1, dim] and one layer's caches."""

    def __init__(self, model: FishOmniTowerModel, k: int) -> None:
        super().__init__(model, "decode")
        self.layer = model.residual_layers[k - 1]
        self.head = model.codebook_heads[k - 1]

    def forward(self, hidden, embed, cos, sin, cache_position, block_tables, k_cache, v_cache):
        hidden = self.layer(hidden, embed, cos, sin, k_cache, v_cache, cache_position, block_tables, self.phase)
        return self.head(hidden), hidden


class FishOmniTowerPrefill(_TowerGraph):
    """Teacher-forced prefill of every layer's cache for one sequence chunk (batch 1).

    Inputs: hidden [1, T, dim], layer_embeds [L, 1, T, dim], cos / sin [1, T, rotary_dim] fp32,
    cache_position [1, 1] int32 (chunk start), block_tables [1] int16, then K, V caches of all L layers.
    Returns the last layer's hidden state [1, T, dim] (upstream prefill only fills the caches).
    """

    def __init__(self, model: FishOmniTowerModel) -> None:
        super().__init__(model, "prefill")
        self.layers = nn.ModuleList(model.layers)

    def forward(self, hidden, layer_embeds, cos, sin, cache_position, block_tables, *kv_caches):
        for index, layer in enumerate(self.layers):
            hidden = layer(
                hidden,
                layer_embeds[index],
                cos,
                sin,
                kv_caches[2 * index],
                kv_caches[2 * index + 1],
                cache_position,
                block_tables,
                self.phase,
            )
        return hidden
