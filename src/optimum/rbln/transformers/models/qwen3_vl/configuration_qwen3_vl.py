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

from ....configuration_utils import RBLNModelConfig
from ....utils.deprecation import deprecate_method
from ..decoderonly.configuration_decoderonly import RBLNDecoderOnlyModelConfig, RBLNDecoderOnlyModelForCausalLMConfig


class DeprecatedLoadVisualRuntimeMixin:
    """Keeps the removed `_load_visual_runtime` knob writable by forwarding it to `visual.create_runtimes`."""

    visual: RBLNModelConfig | dict[str, Any]

    @property
    def _load_visual_runtime(self) -> bool | None:
        if isinstance(self.visual, dict):
            return self.visual.get("create_runtimes")
        return self.visual.create_runtimes

    @_load_visual_runtime.setter
    @deprecate_method(
        version="0.13.0", new_method='visual={"create_runtimes": ...}', raise_if_greater_or_equal_version=False
    )
    def _load_visual_runtime(self, load_visual_runtime: bool) -> None:
        if isinstance(self.visual, dict):
            self.visual["create_runtimes"] = load_visual_runtime
        else:
            self.visual.create_runtimes = load_visual_runtime

    def get_load_overrides(self) -> dict[str, Any]:
        """Leaves out the `_load_visual_runtime` alias, whose value `visual` already carries as `create_runtimes`.

        Returns:
            Dictionary of load overrides, keyed like the ``rbln_config`` dict accepted by ``from_pretrained``.
        """
        overrides = super().get_load_overrides()
        overrides.pop("_load_visual_runtime", None)
        return overrides


class RBLNQwen3VLForConditionalGenerationConfig(
    DeprecatedLoadVisualRuntimeMixin, RBLNDecoderOnlyModelForCausalLMConfig
):
    """
    Configuration class for RBLNQwen3VLForConditionalGeneration.

    This configuration class stores the configuration parameters specific to
    RBLN-optimized Qwen3-VL models for multimodal conditional generation tasks
    that combine vision and language processing capabilities.
    """

    submodules = ["visual"]
    subclass_non_save_attributes = [
        *RBLNDecoderOnlyModelForCausalLMConfig.subclass_non_save_attributes,
        "_load_visual_runtime",
    ]

    def __init__(
        self,
        use_inputs_embeds: bool = True,
        visual: RBLNModelConfig | None = None,
        _load_visual_runtime: bool | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            use_inputs_embeds (bool): Whether or not to use `inputs_embeds` as input. Defaults to `True`.
            visual (RBLNModelConfig | None): Configuration for the vision encoder component.
            _load_visual_runtime (bool | None): Deprecated. Use `visual={"create_runtimes": ...}` instead.
            kwargs: Additional arguments passed to the parent `RBLNDecoderOnlyModelForCausalLMConfig`.

        Raises:
            ValueError: If `use_inputs_embeds` is False.
        """
        super().__init__(use_inputs_embeds=use_inputs_embeds, **kwargs)
        if not self.use_inputs_embeds:
            raise ValueError(
                "RBLNQwen3VLForConditionalGenerationConfig requires use_inputs_embeds=True. "
                "The visual encoder output must be injected into inputs_embeds, whether the "
                "visual encoder runs locally or embeddings are received from an encoder node."
            )
        self.visual = self.initialize_submodule_config(submodule_config=visual, batch_size=1, force_kwargs=True)
        if _load_visual_runtime is not None:
            self._load_visual_runtime = _load_visual_runtime


class RBLNQwen3VLModelConfig(DeprecatedLoadVisualRuntimeMixin, RBLNDecoderOnlyModelConfig):
    submodules = ["visual"]
    subclass_non_save_attributes = [
        *RBLNDecoderOnlyModelConfig.subclass_non_save_attributes,
        "_load_visual_runtime",
    ]

    def __init__(self, visual: RBLNModelConfig | None = None, _load_visual_runtime: bool | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        if not getattr(self, "use_inputs_embeds", True):
            raise ValueError(
                "RBLNQwen3VLModelConfig requires use_inputs_embeds=True. "
                "The visual encoder output must be injected into inputs_embeds, whether the "
                "visual encoder runs locally or embeddings are received from an encoder node."
            )
        self.visual = self.initialize_submodule_config(submodule_config=visual, batch_size=1, force_kwargs=True)
        if _load_visual_runtime is not None:
            self._load_visual_runtime = _load_visual_runtime


class RBLNQwen3VLVisionModelConfig(RBLNModelConfig):
    """
    Configuration class for RBLNQwen3VLVisionModel.

    This configuration class stores the configuration parameters specific to
    RBLN-optimized Qwen3-VL vision transformer models for processing images and videos.
    """

    def __init__(self, max_seq_len: int | list[int] | None = None, batch_size: int | None = None, **kwargs: Any):
        """
        Args:
            max_seq_len (int | list[int] | None): Maximum sequence lengths for Vision
                Transformer attention. Can be an integer or list of integers, each indicating
                the number of patches in a sequence for an image or video. For example, an image
                of 512x512 pixels with patch size 16 yields (512/16) * (512/16) = 1024 patches;
                the ViT counts them before the 2x2 spatial merge, on the size after the processor's
                resize. RBLN optimization runs inference per image, or per temporal patch (2 frames)
                of a video, so set `max_seq_len` to match the maximum expected resolution to reduce
                computation. If not provided, a `ValueError` is raised.
            batch_size (int | None): the vision encoder runs one image at a time (the parent config forces
                this by default), so only `batch_size=1` is supported. Defaults to 1.
            kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If `max_seq_len` is None or not provided.
            ValueError: If `batch_size` is not 1.
        """
        super().__init__(**kwargs)

        batch_size = batch_size or 1
        if batch_size != 1:
            raise ValueError(f"The Qwen3-VL vision encoder only supports batch_size=1, got {batch_size}.")
        self.batch_size = batch_size

        if max_seq_len is not None:
            if isinstance(max_seq_len, int):
                max_seq_len = [max_seq_len]
            elif isinstance(max_seq_len, list):
                max_seq_len.sort(reverse=True)
        else:
            raise ValueError("'max_seq_len' must be specified.")

        self.max_seq_len = max_seq_len


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
    "RBLNQwen3VLForConditionalGenerationConfig",
    "RBLNQwen3VLModelConfig",
    "RBLNQwen3VLTextEncoderConfig",
    "RBLNQwen3VLVisionModelConfig",
]
