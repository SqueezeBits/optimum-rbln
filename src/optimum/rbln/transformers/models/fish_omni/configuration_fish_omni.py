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

from typing import Any

from ..qwen3_5_moe.configuration_qwen3_5_moe import RBLNQwen3_5MoeForCausalLMConfig


class RBLNFishOmniForCausalLMConfig(RBLNQwen3_5MoeForCausalLMConfig):
    """
    Configuration class for the RBLN FishOmni text backbone (causal language model).

    The FishOmni text backbone is the Qwen3.5-MoE hybrid decoder (GatedDeltaNet `linear_attention` layers and gated
    `full_attention` layers, each with a sparse Mixture-of-Experts block plus a gated shared expert). The options are
    the same as `RBLNQwen3_5MoeForCausalLMConfig`.

    Example usage:
    ```python
    from optimum.rbln import RBLNFishOmniForCausalLM, RBLNFishOmniForCausalLMConfig

    config = RBLNFishOmniForCausalLMConfig(
        batch_size=1,
        max_seq_len=8192,
        num_devices=8,
        expert_storage="fp8",
    )
    model = RBLNFishOmniForCausalLM.from_pretrained("fishaudio/drama-3", export=True, rbln_config=config)
    ```
    """

    # A `decoder_batch_<b>` graph reads and writes rows [:b] of the linear-attention state (FishOmniLinearDecoderLayer)
    _allows_decoder_batch_bucketing = True

    def __init__(
        self,
        gdn_chunk_size: int | None = None,
        linear_attention_layers: list[int] | None = None,
        output_last_hidden_state: bool | None = None,
        expert_storage: str | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            gdn_chunk_size (Optional[int]): GatedDeltaNet prefill sub-chunk size, as in `RBLNQwen3_5ForCausalLMConfig`.
            linear_attention_layers (list[int] | None): populated from `layer_types` at compile time (not user-set).
            output_last_hidden_state (bool | None): the graphs also return the final-norm hidden state (every row of
                a prefill chunk, the batch rows of a decode step) as their last output; the runtimes hand it back as
                `hidden_states=(final_hidden,)`. The FishOmni codebook tower reads it. Defaults to False.
            expert_storage (str | None): how the checkpoint's block-FP8 routed experts are loaded and compiled.
                "fp8" keeps them at 8 bits with their block scales and runs `custom_moe_glu_group_dequantize`;
                "bf16" dequantizes them to the rbln dtype while loading and runs `custom_moe_glu`. Defaults to
                "fp8", which is also how an artifact whose rbln_config.json predates this field was compiled.
            kwargs: Additional arguments passed to `RBLNQwen3_5MoeForCausalLMConfig`.

        Unlike Qwen3.5, `decoder_batch_sizes` may hold several sizes (`_allows_decoder_batch_bucketing`): a
        `decoder_batch_<b>` graph reads and writes rows [:b] of the linear-attention state
        (`FishOmniLinearDecoderLayer`), the rows a batch-b prefill fills.

        Raises:
            ValueError: If both `output_last_hidden_state` and `output_hidden_states` are set, or if
                `expert_storage` is not "fp8" or "bf16".
        """
        super().__init__(gdn_chunk_size=gdn_chunk_size, linear_attention_layers=linear_attention_layers, **kwargs)
        self.output_last_hidden_state = bool(output_last_hidden_state)
        if self.output_last_hidden_state and self.output_hidden_states:
            raise ValueError("set output_last_hidden_state or output_hidden_states, not both")
        self.expert_storage = "fp8" if expert_storage is None else expert_storage
        if self.expert_storage not in ("fp8", "bf16"):
            raise ValueError(f'expert_storage must be "fp8" or "bf16", got {self.expert_storage!r}')


__all__ = ["RBLNFishOmniForCausalLMConfig"]
