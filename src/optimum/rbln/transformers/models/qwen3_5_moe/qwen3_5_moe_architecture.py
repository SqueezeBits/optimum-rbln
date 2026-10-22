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

from ...utils.moe import compute_masked_routing_weight_softmax_first, split_fused_experts
from ..qwen3_5.qwen3_5_architecture import Qwen3_5_CausalLMWrapper, Qwen3_5_LanguageModelWrapper


class Qwen3_5MoeMLP(nn.Module):
    def __init__(self, experts: nn.Module, top_k: int):
        super().__init__()
        self.num_experts = experts.num_experts
        self.top_k = top_k

        # Fused Qwen3_5MoeExperts: gate_up_proj [E, 2I, H], down_proj [E, H, I].
        gate, up, down = split_fused_experts(experts)
        self.register_buffer("gate_proj_weight", gate)
        self.register_buffer("up_proj_weight", up)
        self.register_buffer("down_proj_weight", down)

    def forward(self, x: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
        # The HF router always renormalizes the top-k probabilities (softmax -> top-k -> renormalize), which equals a
        # softmax over the top-k logits.
        masked_routing_weight = compute_masked_routing_weight_softmax_first(
            router_logits, top_k=self.top_k, renormalize=True
        )
        return torch.ops.rbln_custom_ops.custom_moe_glu(
            hidden_states=x,
            gate_proj_weight=self.gate_proj_weight,
            up_proj_weight=self.up_proj_weight,
            down_proj_weight=self.down_proj_weight,
            masked_routing_weight=masked_routing_weight,
            hidden_act="silu",
        )


class Qwen3_5MoeSparseMoeBlock(nn.Module):
    """Routed experts on `custom_moe_glu` plus the shared expert scaled by `sigmoid(shared_expert_gate(x))`."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.num_experts = model.gate.num_experts
        self.top_k = model.gate.top_k
        gate_weight = model.gate.weight
        gate = nn.Linear(gate_weight.shape[1], gate_weight.shape[0], bias=False)
        gate.weight = model.gate.weight
        self.gate = gate
        self.experts = Qwen3_5MoeMLP(model.experts, self.top_k)
        self.shared_expert = model.shared_expert
        self.shared_expert_gate = model.shared_expert_gate

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)
        final_hidden_states = self.experts(hidden_states, router_logits)
        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = torch.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output
        final_hidden_states = final_hidden_states + shared_expert_output
        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)


def replace_sparse_moe_blocks(layers: nn.ModuleList) -> None:
    # The Qwen3.5 conversion takes each layer's `mlp` as is, for linear_attention and full_attention layers alike, so
    # swapping the HF MoE block in place is enough to route both layer types through `custom_moe_glu`.
    for layer in layers:
        if isinstance(layer.mlp, HFQwen3_5MoeSparseMoeBlock):
            layer.mlp = Qwen3_5MoeSparseMoeBlock(layer.mlp)


class Qwen3_5Moe_CausalLMWrapper(Qwen3_5_CausalLMWrapper):
    def convert_to_rbln_class(self, model, max_seq_len: int, use_rotary_emb: bool):
        replace_sparse_moe_blocks(self.get_decoder_layers(model))
        return super().convert_to_rbln_class(model, max_seq_len, use_rotary_emb)


class Qwen3_5Moe_LanguageModelWrapper(Qwen3_5_LanguageModelWrapper):
    def convert_to_rbln_class(self, model, max_seq_len: int, use_rotary_emb: bool):
        replace_sparse_moe_blocks(self.get_decoder_layers(model))
        return super().convert_to_rbln_class(model, max_seq_len, use_rotary_emb)
