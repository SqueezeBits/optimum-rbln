# Copyright 2026 Rebellions Inc. All rights reserved.

from typing import Any

from .....configuration_utils import RBLNModelConfig
from .....transformers import RBLNQwen3VLModelConfig, RBLNQwen3VLVisionModelConfig


class RBLNQwen3VLTextEncoderConfig(RBLNModelConfig):
    """
    Configuration class for RBLNQwen3VLTextEncoder.

    The text tower of a Qwen3-VL checkpoint compiled as a single-pass prompt encoder that returns the hidden states
    of selected layers, as used by text-to-image pipelines such as FLUX.3.
    """

    def __init__(
        self,
        batch_size: int | None = None,
        max_seq_len: int | None = None,
        hidden_states_layers: list[int] | None = None,
        exp_silu: bool | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            batch_size (int | None): The batch size for inference. Defaults to 1.
            max_seq_len (int | None): The prompt length the encoder is compiled for. Shorter prompts are padded and
                the added tokens are masked out of attention. Defaults to 1040.
            hidden_states_layers (list[int] | None): Indices into the `hidden_states` returned by the Hugging Face
                model with `output_hidden_states=True` (0 is the embedding output, `i` the output of layer `i`). Only
                these are computed and returned; the layers after the last one are not compiled. Defaults to every
                layer.
            exp_silu (bool | None): On RBLN-CR13, compute the SiLU of the MLPs in the exp form (`replace_silu_for_npu`),
                which keeps the precision the fused SiLU kernel loses. Defaults to False.
            kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If batch_size is not a positive integer.
        """
        super().__init__(**kwargs)
        self.batch_size = batch_size or 1
        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")

        self.max_seq_len = max_seq_len
        self.hidden_states_layers = sorted(hidden_states_layers) if hidden_states_layers is not None else None
        self.exp_silu = bool(exp_silu)


class RBLNQwenImage21VisionModelConfig(RBLNQwen3VLVisionModelConfig):
    """
    Configuration class for RBLNQwenImage21VisionModel: the Qwen3-VL vision transformer of the Qwen-Image 2.1 prompt
    encoder, with the RBLN-CR13 precision option of its activation.
    """

    def __init__(self, exp_gelu: bool | None = None, **kwargs: Any):
        """
        Args:
            exp_gelu (bool | None): On RBLN-CR13, compute the tanh GELU of the block MLPs in the exp form
                (`replace_gelu_tanh_for_npu`), which keeps the precision the tanh kernel loses. Defaults to False.
            kwargs: Additional arguments passed to the parent RBLNQwen3VLVisionModelConfig.
        """
        super().__init__(**kwargs)
        self.exp_gelu = bool(exp_gelu)


class RBLNQwenImage21TextEncoderConfig(RBLNQwen3VLModelConfig):
    """
    Configuration class for RBLNQwenImage21TextEncoder: a Qwen3-VL model as the prompt encoder of Qwen-Image 2.1,
    with the options that encoder needs on top of RBLNQwen3VLModelConfig.
    """

    def __init__(
        self,
        visual: RBLNModelConfig | dict[str, Any] | None = None,
        final_norm: bool | None = None,
        exp_silu: bool | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            visual (RBLNModelConfig | dict | None): Configuration for the vision encoder submodule, an
                RBLNQwenImage21VisionModelConfig unless another `cls_name` is given.
            final_norm (bool | None): Whether the output goes through the final RMSNorm of the language model.
                Qwen-Image reads the last decoder layer's output before it. Defaults to True.
            exp_silu (bool | None): On RBLN-CR13, compute the SiLU of the MLPs in the exp form (`replace_silu_for_npu`),
                which keeps the precision the fused SiLU kernel loses. Defaults to False.
            kwargs: Additional arguments passed to the parent RBLNQwen3VLModelConfig.
        """
        if visual is None:
            visual = {}
        if isinstance(visual, dict):
            visual = {"cls_name": "RBLNQwenImage21VisionModelConfig", **visual}
        super().__init__(visual=visual, **kwargs)
        self.final_norm = True if final_norm is None else final_norm
        self.exp_silu = bool(exp_silu)


__all__ = [
    "RBLNQwen3VLTextEncoderConfig",
    "RBLNQwenImage21TextEncoderConfig",
    "RBLNQwenImage21VisionModelConfig",
]
