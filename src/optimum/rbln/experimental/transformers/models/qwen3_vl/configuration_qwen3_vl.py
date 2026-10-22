# Copyright 2026 Rebellions Inc. All rights reserved.

from typing import Any

from .....configuration_utils import RBLNModelConfig


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


__all__ = [
    "RBLNQwen3VLTextEncoderConfig",
]
