# Copyright 2026 Rebellions Inc. All rights reserved.

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch
from transformers import PretrainedConfig, Qwen3VLForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding

from .....configuration_utils import RBLNCompileConfig
from .....modeling import RBLNModel
from .....modeling_base import Preprocessor
from .....transformers import RBLNQwen3VLModel, RBLNQwen3VLVisionModel
from .....utils.logging import get_logger
from ....modeling_activation_utils import replace_gelu_tanh_for_npu, replace_silu_for_npu
from ....utils import experimental
from .configuration_qwen3_vl import (
    RBLNQwen3VLTextEncoderConfig,
    RBLNQwenImage21TextEncoderConfig,
    RBLNQwenImage21VisionModelConfig,
)
from .qwen3_vl_architecture import Qwen3VLTextEncoderWrapper


if TYPE_CHECKING:
    from transformers import PreTrainedModel


logger = get_logger(__name__)


@experimental
class RBLNQwen3VLTextEncoder(RBLNModel):
    """
    The text tower of a Qwen3-VL checkpoint as a single-pass prompt encoder on RBLN devices.

    Text-to-image pipelines such as FLUX.3 call a `Qwen3VLForConditionalGeneration` once per prompt with
    `output_hidden_states=True` and read a few intermediate layers, including at the padding positions. This class
    compiles exactly that: one graph without a KV cache that runs the decoder layers over the whole padded prompt
    (causal attention over the valid tokens, as with the Hugging Face attention mask) and returns the requested
    `hidden_states` entries. It is a drop-in replacement for that call; it does not generate.

    This class inherits from [`RBLNModel`]. Check the superclass documentation for the generic methods
    the library implements for all its models.
    """

    _hf_class = Qwen3VLForConditionalGeneration
    _rbln_config_class = RBLNQwen3VLTextEncoderConfig

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(self.config.text_config)

    @classmethod
    def update_rbln_config_using_pipe(cls, pipe: Any, rbln_config: Any, submodule_name: str) -> Any:
        # Diffusion pipeline configs set the prompt length and the layers they read.
        return rbln_config

    @classmethod
    def _wrap_model_if_needed(cls, model: "PreTrainedModel", rbln_config: RBLNQwen3VLTextEncoderConfig):
        if rbln_config.exp_silu:
            for layer in model.model.language_model.layers:
                replace_silu_for_npu(layer.mlp, rbln_config.npu)
        return Qwen3VLTextEncoderWrapper(model, rbln_config.hidden_states_layers).eval()

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Sequence[Preprocessor] | None,
        model: "PreTrainedModel",
        model_config: PretrainedConfig,
        rbln_config: RBLNQwen3VLTextEncoderConfig,
    ) -> RBLNQwen3VLTextEncoderConfig:
        text_config = model_config.text_config
        if rbln_config.max_seq_len is None:
            rbln_config.max_seq_len = 1040
        if rbln_config.hidden_states_layers is None:
            rbln_config.hidden_states_layers = list(range(text_config.num_hidden_layers + 1))
        if not 0 <= rbln_config.hidden_states_layers[-1] <= text_config.num_hidden_layers:
            raise ValueError(
                f"hidden_states_layers must be in [0, {text_config.num_hidden_layers}], "
                f"got {rbln_config.hidden_states_layers}."
            )

        shape = [rbln_config.batch_size, rbln_config.max_seq_len]
        head_dim = text_config.head_dim
        input_info = [
            ("input_ids", shape, "int64"),
            ("attention_mask", shape, rbln_config.dtype),
            ("position_cos", [*shape, head_dim], rbln_config.dtype),
            ("position_sin", [*shape, head_dim], rbln_config.dtype),
        ]
        rbln_config.set_compile_cfgs([RBLNCompileConfig(input_info=input_info)])
        return rbln_config

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs: Any,
    ) -> BaseModelOutputWithPast | tuple:
        """
        Encodes a batch of prompts and returns the hidden states of the compiled layers.

        Args:
            input_ids (torch.LongTensor): Token ids of shape `(batch_size, seq_len)`, `seq_len <= max_seq_len`.
            attention_mask (torch.Tensor | None): 1 for the tokens to attend to, 0 for padding. Defaults to all ones.
            position_ids (torch.LongTensor | None): Text position ids of shape `(batch_size, seq_len)`. Defaults to
                `0..seq_len-1`, like the Hugging Face model.
            output_hidden_states (bool | None): Accepted for compatibility; the hidden states are always returned.
            return_dict (bool | None): Whether to return a `BaseModelOutputWithPast` instead of a plain tuple.
            kwargs: Other arguments of the Hugging Face forward (e.g. `use_cache`), ignored.

        Returns:
            `BaseModelOutputWithPast` (or a tuple) whose `hidden_states` has `num_hidden_layers + 1` entries, like the
            Hugging Face output; the entries outside `hidden_states_layers` are `None`.
        """
        batch_size, seq_len = input_ids.shape
        max_seq_len = self.rbln_config.max_seq_len
        if batch_size != self.rbln_config.batch_size:
            raise ValueError(
                f"The encoder was compiled for a batch of {self.rbln_config.batch_size}, got {batch_size}. "
                "Recompile with `batch_size` set to the batch of the call."
            )
        if seq_len > max_seq_len:
            raise ValueError(
                f"The prompt has {seq_len} tokens, but the encoder was compiled for at most {max_seq_len} "
                "(`max_seq_len`). Recompile with a larger `max_seq_len`."
            )
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if seq_len == max_seq_len and bool(attention_mask.all()):
            # Pipelines truncate prompts at the compiled length, so a prompt that fills it was likely cut.
            logger.warning(
                f"The prompt fills all {max_seq_len} tokens the encoder was compiled for and may have been truncated. "
                "Recompile with a larger `max_seq_len` to encode longer prompts."
            )
        if position_ids is None:
            position_ids = torch.arange(seq_len).expand(batch_size, -1)

        # The positions added to reach the compiled length are masked out as keys and cut from the output.
        pad = max_seq_len - seq_len
        input_ids = torch.nn.functional.pad(input_ids, (0, pad))
        attention_mask = torch.nn.functional.pad(attention_mask, (0, pad))
        position_ids = torch.nn.functional.pad(position_ids, (0, pad), value=0)
        # The rotary module reads only the device and dtype of its first argument.
        cos, sin = self.rotary_emb(torch.empty(0, dtype=torch.float32), position_ids)

        dtype = self.rbln_config.dtype
        output = self.model[0](input_ids, attention_mask.to(dtype), cos.to(dtype), sin.to(dtype))
        layers = self.rbln_config.hidden_states_layers
        per_layer = output[:, :seq_len].chunk(len(layers), dim=-1)
        hidden_states = [None] * (self.config.text_config.num_hidden_layers + 1)
        for layer, states in zip(layers, per_layer, strict=True):
            hidden_states[layer] = states
        hidden_states = tuple(hidden_states)

        return_dict = return_dict if return_dict is not None else self.config.return_dict
        if not return_dict:
            return (None, None, hidden_states)
        return BaseModelOutputWithPast(hidden_states=hidden_states)


class RBLNQwenImage21VisionModel(RBLNQwen3VLVisionModel):
    """
    The Qwen3-VL vision transformer of the Qwen-Image 2.1 prompt encoder: RBLNQwen3VLVisionModel with the block
    activations rewritten for RBLN-CR13 (`exp_gelu`).
    """

    @classmethod
    def _wrap_model_if_needed(cls, model: "PreTrainedModel", rbln_config: RBLNQwenImage21VisionModelConfig):
        if rbln_config.exp_gelu:
            for vision_block in model.blocks:
                replace_gelu_tanh_for_npu(vision_block.mlp, rbln_config.npu)
        return super()._wrap_model_if_needed(model, rbln_config)


@experimental
class RBLNQwenImage21TextEncoder(RBLNQwen3VLModel):
    """
    A Qwen3-VL model as the prompt encoder of Qwen-Image 2.1: RBLNQwen3VLModel compiled from the pipeline's
    `Qwen3VLForConditionalGeneration`, with the MLP activations rewritten for RBLN-CR13 (`exp_silu`) and the last
    layer's output returned before the final norm (`final_norm=False`), as the pipeline reads it.
    """

    _hf_class = Qwen3VLForConditionalGeneration
    _rbln_submodules = [{"name": "visual", "config_class": RBLNQwenImage21VisionModelConfig}]

    @classmethod
    def update_rbln_config_using_pipe(cls, pipe: Any, rbln_config: Any, submodule_name: str) -> Any:
        # As a pipeline's prompt encoder, the pipeline config sets everything.
        return rbln_config

    @classmethod
    def _reconstruct_model_if_needed(cls, model: "PreTrainedModel"):
        # A pipeline hands over the Qwen3VLForConditionalGeneration it loaded: compile its base model (no LM head).
        return model.model if getattr(model, "lm_head", None) is not None else model

    @classmethod
    def _wrap_model_if_needed(cls, model: "PreTrainedModel", rbln_config: RBLNQwenImage21TextEncoderConfig):
        # The decoder wrapper takes the layers' MLPs and the final norm as they are, so they are changed here first.
        language_model = model.get_decoder()
        if rbln_config.exp_silu:
            for layer in language_model.layers:
                replace_silu_for_npu(layer.mlp, rbln_config.npu)
        if not rbln_config.final_norm:
            language_model.norm = torch.nn.Identity()
        return super()._wrap_model_if_needed(model, rbln_config)


__all__ = [
    "RBLNQwen3VLTextEncoder",
    "RBLNQwenImage21TextEncoder",
    "RBLNQwenImage21VisionModel",
]
