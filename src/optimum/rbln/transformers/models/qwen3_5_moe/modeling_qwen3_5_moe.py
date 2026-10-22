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

from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeModel, Qwen3_5MoeTextRotaryEmbedding

from ...utils.moe import RBLNMoeLoadMixin
from ..qwen3_5.modeling_qwen3_5 import (
    RBLNQwen3_5ForCausalLM,
    RBLNQwen3_5ForConditionalGeneration,
    RBLNQwen3_5Model,
    RBLNQwen3_5TextModel,
    RBLNQwen3_5VisionModel,
)
from .qwen3_5_moe_architecture import Qwen3_5Moe_CausalLMWrapper, Qwen3_5Moe_LanguageModelWrapper


class RBLNQwen3_5MoeTextModel(RBLNMoeLoadMixin, RBLNQwen3_5TextModel):
    """The bare Qwen3.5-MoE text backbone (no LM head).

    The hybrid Qwen3.5 decoder (`linear_attention` GatedDeltaNet layers and gated `full_attention` layers) with a
    sparse Mixture-of-Experts block plus a gated shared expert in place of every dense MLP. Everything but the MoE
    block is shared with `RBLNQwen3_5TextModel`.
    """

    _decoder_wrapper_cls = Qwen3_5Moe_CausalLMWrapper


class RBLNQwen3_5MoeForCausalLM(RBLNMoeLoadMixin, RBLNQwen3_5ForCausalLM):
    """
    RBLNQwen3_5MoeForCausalLM is the text-only (causal language modeling) variant of Qwen3.5-MoE, optimized for RBLN NPUs.
    It runs the hybrid Qwen3.5 decoder, GatedDeltaNet `linear_attention` layers interleaved with gated
    `full_attention` layers, where every layer's feed-forward network is a sparse Mixture-of-Experts block plus a
    shared expert scaled by a sigmoid gate. The vision encoder is not used.

    This model inherits from [`RBLNDecoderOnlyModelForCausalLM`]. Check the superclass documentation for the generic methods the library implements for all its models.

    **Configuration:**
    This model uses [`RBLNQwen3_5MoeForCausalLMConfig`] for configuration. When calling methods like `from_pretrained` or `from_model`,
    the `rbln_config` parameter should be an instance of [`RBLNQwen3_5MoeForCausalLMConfig`] or a dictionary conforming to its structure.
    See the [`RBLNQwen3_5MoeForCausalLMConfig`] class for all available configuration options.

    Examples:
        ```python
        from optimum.rbln import RBLNQwen3_5MoeForCausalLM

        # Simple usage using rbln_* arguments
        model = RBLNQwen3_5MoeForCausalLM.from_pretrained(
            "Qwen/Qwen3.5-35B-A3B",
            export=True,
            rbln_max_seq_len=32768,
            rbln_num_devices=8,
        )

        # Using a config dictionary
        rbln_config = {
            "batch_size": 1,
            "max_seq_len": 32768,
            "num_devices": 8,
        }
        model = RBLNQwen3_5MoeForCausalLM.from_pretrained(
            "Qwen/Qwen3.5-35B-A3B",
            export=True,
            rbln_config=rbln_config,
        )

        # Using a RBLNQwen3_5MoeForCausalLMConfig instance (recommended for type checking)
        from optimum.rbln import RBLNQwen3_5MoeForCausalLMConfig

        config = RBLNQwen3_5MoeForCausalLMConfig(
            batch_size=1,
            max_seq_len=32768,
            num_devices=8,
        )
        model = RBLNQwen3_5MoeForCausalLM.from_pretrained(
            "Qwen/Qwen3.5-35B-A3B",
            export=True,
            rbln_config=config,
        )
        ```
    """

    _decoder_wrapper_cls = Qwen3_5Moe_CausalLMWrapper


class RBLNQwen3_5MoeVisionModel(RBLNQwen3_5VisionModel):
    """Qwen3.5-MoE vision encoder for RBLN.

    transformers generates the Qwen3.5-MoE vision classes as copies of the Qwen3.5 ones, so the Qwen3.5 encoder
    (no deepstack) runs it unchanged.
    """


class RBLNQwen3_5MoeModel(RBLNMoeLoadMixin, RBLNQwen3_5Model):
    """The bare Qwen3.5-MoE model (vision encoder + hybrid MoE text backbone, no LM head).

    Qwen3.5-MoE is natively vision-language. The vision encoder output is injected into `inputs_embeds`; see
    `RBLNQwen3_5Model` for the runtime, which this class shares except for the MoE text layers.
    """

    _decoder_wrapper_cls = Qwen3_5Moe_LanguageModelWrapper
    _config_class = Qwen3_5MoeConfig
    _rotary_emb_class = Qwen3_5MoeTextRotaryEmbedding
    _get_rope_index_func = Qwen3_5MoeModel.get_rope_index
    get_vision_position_ids = Qwen3_5MoeModel.get_vision_position_ids


class RBLNQwen3_5MoeForConditionalGeneration(RBLNMoeLoadMixin, RBLNQwen3_5ForConditionalGeneration):
    """
    RBLNQwen3_5MoeForConditionalGeneration is a multi-modal model that integrates vision and language processing capabilities,
    optimized for RBLN NPUs. It pairs the Qwen3.5 vision encoder with the hybrid Qwen3.5-MoE text backbone, where
    every layer's feed-forward network is a sparse Mixture-of-Experts block plus a gated shared expert.

    This model inherits from [`RBLNDecoderOnlyModelForCausalLM`]. Check the superclass documentation for the generic methods the library implements for all its models.

    **Configuration:**
    This model uses [`RBLNQwen3_5MoeForConditionalGenerationConfig`] for configuration. When calling methods like `from_pretrained` or `from_model`,
    the `rbln_config` parameter should be an instance of [`RBLNQwen3_5MoeForConditionalGenerationConfig`] or a dictionary conforming to its structure.
    See the [`RBLNQwen3_5MoeForConditionalGenerationConfig`] class for all available configuration options.

    Examples:
        ```python
        from optimum.rbln import RBLNQwen3_5MoeForConditionalGeneration

        model = RBLNQwen3_5MoeForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3.5-35B-A3B",
            export=True,
            rbln_config={
                "visual": {
                    "num_devices": 8,
                    "max_seq_len": 16384,
                    "device": [0, 1, 2, 3, 4, 5, 6, 7],
                },
                "num_devices": 8,
                "kvcache_partition_len": 16384,
                "max_seq_len": 262144,
                "device": [0, 1, 2, 3, 4, 5, 6, 7],
            },
        )

        model.save_pretrained("qwen3.5-35B-A3B")
        ```
    """

    _decoder_wrapper_cls = Qwen3_5Moe_LanguageModelWrapper
    _config_class = Qwen3_5MoeConfig
    _rotary_emb_class = Qwen3_5MoeTextRotaryEmbedding
    _get_rope_index_func = Qwen3_5MoeModel.get_rope_index
    get_vision_position_ids = Qwen3_5MoeModel.get_vision_position_ids


__all__ = [
    "RBLNQwen3_5MoeForCausalLM",
    "RBLNQwen3_5MoeForConditionalGeneration",
    "RBLNQwen3_5MoeModel",
    "RBLNQwen3_5MoeTextModel",
    "RBLNQwen3_5MoeVisionModel",
]
