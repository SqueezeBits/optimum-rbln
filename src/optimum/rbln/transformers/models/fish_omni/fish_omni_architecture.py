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

import torch
from torch import nn
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeSparseMoeBlock as HFQwen3_5MoeSparseMoeBlock,
)

from ...utils.moe import compute_masked_routing_weight_softmax_first
from ..qwen3_5.qwen3_5_architecture import Qwen3_5ForCausalLM, Qwen3_5LinearDecoderLayer
from ..qwen3_5_moe.qwen3_5_moe_architecture import Qwen3_5Moe_CausalLMWrapper, Qwen3_5MoeSparseMoeBlock


FP8_BLOCK_SIZE = 128
# Buffer names the loader registers on an HF experts module whose weights stay block FP8.
GATE_UP_SCALE = "gate_up_proj_scale_inv"
DOWN_SCALE = "down_proj_scale_inv"


class FishOmniFp8MoeMLP(nn.Module):
    """Routed experts kept in block FP8 and run through `custom_moe_glu_group_dequantize`.

    The HF experts module holds `gate_up_proj` [E, 2I, H] and `down_proj` [E, H, I] in float8_e4m3fn with
    `weight_scale_inv` blocks [E, 2I/128, H/128] and [E, H/128, I/128]. The op takes each weight as
    [E, out, in] with scale [E, out_blocks, in_blocks] (its CPU implementation; the docstring layout differs).
    """

    def __init__(self, experts: nn.Module, top_k: int, block_size: int = FP8_BLOCK_SIZE):
        super().__init__()
        self.num_experts = experts.num_experts
        self.top_k = top_k
        self.block_size = block_size

        gate_up = experts.get_parameter("gate_up_proj").detach()
        gate_up_scale = experts.get_buffer(GATE_UP_SCALE)
        intermediate = gate_up.shape[1] // 2
        if intermediate % block_size:
            raise ValueError(f"moe_intermediate_size {intermediate} is not a multiple of the FP8 block {block_size}")
        blocks = intermediate // block_size
        self.register_buffer("gate_proj_weight", gate_up[:, :intermediate].contiguous())
        self.register_buffer("up_proj_weight", gate_up[:, intermediate:].contiguous())
        self.register_buffer("gate_proj_scale", gate_up_scale[:, :blocks].contiguous())
        self.register_buffer("up_proj_scale", gate_up_scale[:, blocks:].contiguous())
        self.register_buffer("down_proj_weight", experts.get_parameter("down_proj").detach())
        self.register_buffer("down_proj_scale", experts.get_buffer(DOWN_SCALE))
        # The HF experts module is not run once wrapped; drop the fused copy so host memory is not doubled.
        experts.register_parameter("gate_up_proj", None)

    def forward(self, x: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
        masked_routing_weight = compute_masked_routing_weight_softmax_first(
            router_logits, top_k=self.top_k, renormalize=True
        )
        return torch.ops.rbln_custom_ops.custom_moe_glu_group_dequantize(
            hidden_states=x,
            gate_proj_weight=self.gate_proj_weight,
            gate_proj_scale=self.gate_proj_scale,
            up_proj_weight=self.up_proj_weight,
            up_proj_scale=self.up_proj_scale,
            down_proj_weight=self.down_proj_weight,
            down_proj_scale=self.down_proj_scale,
            masked_routing_weight=masked_routing_weight,
            group_size=torch.tensor(self.block_size, dtype=torch.int32),
            hidden_act="silu",
        )


class FishOmniFp8SparseMoeBlock(Qwen3_5MoeSparseMoeBlock):
    """`Qwen3_5MoeSparseMoeBlock` with the routed experts on `FishOmniFp8MoeMLP`; router and shared expert unchanged."""

    _experts_cls = FishOmniFp8MoeMLP


def replace_fp8_sparse_moe_blocks(layers: nn.ModuleList) -> None:
    # Only blocks whose experts the loader kept in FP8 are swapped; bf16 experts fall through to the parent wrapper.
    for layer in layers:
        mlp = layer.mlp
        if isinstance(mlp, HFQwen3_5MoeSparseMoeBlock) and GATE_UP_SCALE in mlp.experts._buffers:
            layer.mlp = FishOmniFp8SparseMoeBlock(mlp)


class FishOmniLinearDecoderLayer(Qwen3_5LinearDecoderLayer):
    """`Qwen3_5LinearDecoderLayer` that serves a decode batch smaller than the linear-attention state cache.

    The conv/recurrent caches are sized to the runtime batch. A `decoder_batch_<b>` graph with b below it reads rows
    [:b], the rows a batch-b prefill writes (batch_idx 0..b-1) and the rows the paged KV cache serves it
    (block_tables[:b]); the parent model's `rbln_cache_update` at row 0 writes those b rows back. Prefill (one row,
    already gathered by batch_idx) and the full-batch decoder pass through unchanged.
    """

    def forward(
        self, hidden_states: torch.Tensor, conv_state: torch.Tensor, recurrent_state: torch.Tensor, *args, **kwargs
    ):
        batch = hidden_states.shape[0]
        if conv_state.shape[0] != batch:
            conv_state, recurrent_state = conv_state[:batch], recurrent_state[:batch]
        return super().forward(hidden_states, conv_state, recurrent_state, *args, **kwargs)


class FishOmniForCausalLM(Qwen3_5ForCausalLM):
    """`Qwen3_5ForCausalLM` that, with `output_last_hidden_state`, also hands back the final-norm hidden state, the
    input of `lm_head`, through the parent's `_build_outputs` hook. The forward and the logits path are the parent's.

    The FishOmni codebook tower reads that hidden for every prompt row and every decode step. It is taken before the
    prefill's `query_position` slice, so prefill returns every row of the chunk and decode the batch rows. The wrapper
    puts it after the linear-attention states: an output placed before them keeps every linear layer's conv input and
    state alive to the graph end (about 11.5 MiB more per linear layer).
    """

    def _build_outputs(self, logits, new_states, all_hidden_states, final_hidden):
        outputs = super()._build_outputs(logits, new_states, all_hidden_states, final_hidden)
        if self.model.rbln_config.output_last_hidden_state:
            outputs += (final_hidden,)
        return outputs


class FishOmni_CausalLMWrapper(Qwen3_5Moe_CausalLMWrapper):
    """FishOmni text backbone wrapper.

    Experts loaded with `rbln_config.expert_storage="fp8"` (the default) keep their block-FP8 weights and run on
    `custom_moe_glu_group_dequantize`; experts loaded with `expert_storage="bf16"` are dequantized at load time and
    take the parent's `custom_moe_glu` path. Linear-attention layers become `FishOmniLinearDecoderLayer` so
    `decoder_batch_sizes` may hold several sizes. With `output_last_hidden_state` the graphs also return the
    final-norm hidden as their last output (`FishOmniForCausalLM`).
    """

    def get_rbln_causal_lm_class(self):
        return FishOmniForCausalLM

    def get_rbln_linear_layer_class(self):
        return FishOmniLinearDecoderLayer

    def convert_to_rbln_class(self, model, max_seq_len: int, use_rotary_emb: bool):
        replace_fp8_sparse_moe_blocks(self.get_decoder_layers(model))
        return super().convert_to_rbln_class(model, max_seq_len, use_rotary_emb)
