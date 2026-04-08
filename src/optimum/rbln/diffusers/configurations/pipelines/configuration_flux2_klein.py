from typing import Any, Optional, Tuple

from ....configuration_utils import RBLNModelConfig
from ....transformers import RBLNQwen3ForCausalLMConfig
from ..models import RBLNAutoencoderKLFlux2Config


class RBLNFlux2KleinPipelineConfig(RBLNModelConfig):
    submodules = ["text_encoder", "vae"]

    def __init__(
        self,
        text_encoder: Optional[RBLNQwen3ForCausalLMConfig] = None,
        vae: Optional[RBLNAutoencoderKLFlux2Config] = None,
        *,
        batch_size: Optional[int] = None,
        image_size: Optional[Tuple[int, int]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        max_sequence_length: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        if image_size is not None and (height is not None or width is not None):
            raise ValueError("image_size cannot be provided alongside height/width")

        if (height is None) != (width is None):
            raise ValueError("Both height and width must be provided together if used")

        if image_size is None and height is not None and width is not None:
            image_size = (height, width)

        if image_size is None:
            image_size = (1024, 1024)

        if max_seq_len is not None and max_sequence_length is not None and max_seq_len != max_sequence_length:
            raise ValueError("max_seq_len and max_sequence_length must match when both are provided")

        max_seq_len = max_seq_len or max_sequence_length or 512

        self.text_encoder = self.initialize_submodule_config(
            text_encoder,
            cls_name="RBLNQwen3ForCausalLMConfig",
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            use_attention_mask=True,
            output_hidden_states=True,
            phases=["prefill"],
        )
        self.vae = self.initialize_submodule_config(
            vae,
            cls_name="RBLNAutoencoderKLFlux2Config",
            batch_size=batch_size,
            uses_encoder=True,
            sample_size=image_size,
        )

    @property
    def batch_size(self):
        return self.text_encoder.batch_size

    @property
    def max_seq_len(self):
        return self.text_encoder.max_seq_len

    @property
    def image_size(self):
        return self.vae.sample_size
