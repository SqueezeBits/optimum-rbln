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

"""The Qwen-Image 2.1 Turbo modular pipeline on RBLN NPUs, on the diffusers blocks.

The blocks of `QwenImage21TurboAutoBlocks` run as they are, except three that are swapped at construction:
- the prompt-encoder steps (`QwenImage21TurboTextEncoderStep`, `QwenImage21TurboVLTextEncoderStep`): the original's
  `get_qwenimage21_prompt_embeds` hooks the final norm of the Hugging Face model, which a compiled encoder has no
  equivalent of, so `get_qwenimage21_prompt_embeds` here encodes with `RBLNQwenImage21TextEncoder`;
- the latents step (`QwenImage21PrepareLatentsStep`): the initial noise is drawn in bfloat16, so that a seed gives
  the image of the original pipeline in bfloat16 while the prompt embeddings stay float32.
"""

import math
from typing import Any

import numpy as np
import PIL.Image
import PIL.ImageOps
import torch
from diffusers.modular_pipelines.modular_pipeline import PipelineState
from diffusers.modular_pipelines.qwenimage21 import QwenImage21TurboAutoBlocks, QwenImage21TurboModularPipeline
from diffusers.modular_pipelines.qwenimage21.before_denoise import QwenImage21PrepareLatentsStep
from diffusers.modular_pipelines.qwenimage21.encoders import (
    QWENIMAGE21_IMAGE_TEMPLATE,
    QWENIMAGE21_SYSTEM_PROMPT,
    QwenImage21TurboTextEncoderStep,
    QwenImage21TurboVLTextEncoderStep,
)
from diffusers.modular_pipelines.qwenimage21.inputs import pack_latents
from diffusers.utils.torch_utils import randn_tensor

from ....utils import experimental
from ...configurations import RBLNQwenImage21TurboModularPipelineConfig
from ...configurations.pipelines.configuration_qwenimage21 import QWENIMAGE21_CONDITION_RESOLUTION, qwenimage21_size
from ...modeling_modular_diffusers import RBLNModularPipelineMixin


def _pad_rows(tensor: torch.Tensor, length: int) -> torch.Tensor:
    return torch.cat([tensor, tensor.new_zeros((length - tensor.shape[0], *tensor.shape[1:]))])


def _over_white(image: PIL.Image.Image) -> PIL.Image.Image:
    # As the original: the vision encoder reads RGBA condition images composited over white.
    if image.mode != "RGBA":
        return image
    white = PIL.Image.new("RGB", image.size, (255, 255, 255))
    white.paste(image, mask=image.getchannel("A"))
    return white


def get_qwenimage21_prompt_embeds(
    text_encoder: Any,
    processor: Any,
    prompt: str | list[str],
    image: list[PIL.Image.Image] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mirrors the diffusers `get_qwenimage21_prompt_embeds` with the compiled prompt encoder: the prompts are
    encoded one at a time (the encoder is compiled for one sequence, and each prompt then gets the positions it has in
    the original's left-padded batch), and the encoder returns the last decoder layer's output before the final norm
    (`final_norm=False`), which the original gets with a hook on that norm."""
    prompt = [prompt] if isinstance(prompt, str) else prompt
    # Qwen has no bos token, so an empty string leaves the encoder with nothing to read.
    prompt = [p if p else " " for p in prompt]
    sys_message = [{"role": "system", "content": [{"type": "text", "text": QWENIMAGE21_SYSTEM_PROMPT}]}]
    drop_idx = len(processor.apply_chat_template(sys_message, tokenize=True, return_dict=False)[0])
    img_token_id = processor.tokenizer.encode("<|image_pad|>")[0]

    image_prompt, processor_kwargs = "", {}
    if image is not None:
        image_prompt = " ".join(QWENIMAGE21_IMAGE_TEMPLATE.format(i + 1) for i in range(len(image)))
        processor_kwargs["images"] = [_over_white(img) for img in image]
    template = (
        f"<|im_start|>system\n{QWENIMAGE21_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{image_prompt}{{}}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    embeds, image_pad_masks = [], []
    for text in prompt:
        inputs = processor(text=[template.format(text)], return_tensors="pt", **processor_kwargs)
        encoder_inputs = {
            key: getattr(inputs, key)
            for key in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids")
            if getattr(inputs, key, None) is not None
        }
        hidden_states = text_encoder(**encoder_inputs, return_dict=True).logits[0]
        embeds.append(hidden_states[drop_idx:])
        image_pad_masks.append((inputs.input_ids[0] == img_token_id)[drop_idx:])

    max_seq_len = max(e.shape[0] for e in embeds)
    prompt_embeds = torch.stack([_pad_rows(e, max_seq_len) for e in embeds])
    encoder_attention_mask = torch.stack(
        [_pad_rows(torch.ones(e.shape[0], dtype=torch.long), max_seq_len) for e in embeds]
    )
    image_pad_mask = torch.stack([_pad_rows(m, max_seq_len) for m in image_pad_masks])
    return prompt_embeds, encoder_attention_mask, image_pad_mask


class _TurboTextEncoderStep(QwenImage21TurboTextEncoderStep):
    """`QwenImage21TurboTextEncoderStep` encoding with the compiled prompt encoder."""

    @torch.no_grad()
    def __call__(self, components: Any, state: PipelineState) -> tuple[Any, PipelineState]:
        block_state = self.get_block_state(state)
        self.check_inputs(block_state.prompt)
        block_state.prompt_embeds, block_state.prompt_embeds_mask, block_state.image_pad_mask = (
            get_qwenimage21_prompt_embeds(components.text_encoder, components.processor, prompt=block_state.prompt)
        )
        self.set_block_state(state, block_state)
        return components, state


class _TurboVLTextEncoderStep(QwenImage21TurboVLTextEncoderStep):
    """`QwenImage21TurboVLTextEncoderStep` encoding with the compiled prompt encoder."""

    @torch.no_grad()
    def __call__(self, components: Any, state: PipelineState) -> tuple[Any, PipelineState]:
        block_state = self.get_block_state(state)
        self.check_inputs(block_state.prompt)
        block_state.prompt_embeds, block_state.prompt_embeds_mask, block_state.image_pad_mask = (
            get_qwenimage21_prompt_embeds(
                components.text_encoder,
                components.processor,
                prompt=block_state.prompt,
                image=block_state.resized_image,
            )
        )
        self.set_block_state(state, block_state)
        return components, state


class _PrepareLatentsStep(QwenImage21PrepareLatentsStep):
    """`QwenImage21PrepareLatentsStep` with the initial noise drawn in bfloat16, as by the original pipeline in
    bfloat16, so that a seed gives the same image; the loop then runs in the dtype of the prompt embeddings (float32)
    as in the original."""

    @torch.no_grad()
    def __call__(self, components: Any, state: PipelineState) -> tuple[Any, PipelineState]:
        block_state = self.get_block_state(state)
        if block_state.latents is None:
            multiple_of = components.vae_scale_factor * 2
            height = (block_state.height or block_state.output_resolution) // multiple_of * multiple_of
            width = (block_state.width or block_state.output_resolution) // multiple_of * multiple_of
            shape = (
                block_state.batch_size * block_state.num_images_per_prompt,
                components.num_channels_latents,
                1,
                height // components.vae_scale_factor,
                width // components.vae_scale_factor,
            )
            noise = randn_tensor(shape, generator=block_state.generator, dtype=torch.bfloat16)
            block_state.latents = pack_latents(noise)
            self.set_block_state(state, block_state)
        return super().__call__(components, state)


def rbln_turbo_blocks() -> QwenImage21TurboAutoBlocks:
    """`QwenImage21TurboAutoBlocks` with the steps that call the prompt encoder and the one that draws the initial
    noise replaced; the composition of the blocks is the diffusers one."""
    blocks = QwenImage21TurboAutoBlocks()
    text_encoder = blocks.sub_blocks["text_encoder"]
    text_encoder.sub_blocks["text2image"] = _TurboTextEncoderStep()
    text_encoder.sub_blocks["image_conditioned"].sub_blocks["encode"] = _TurboVLTextEncoderStep()
    for name in ("text2image", "image_conditioned"):
        blocks.sub_blocks["denoise"].sub_blocks[name].sub_blocks["prepare_latents"] = _PrepareLatentsStep()
    return blocks


@experimental
class RBLNQwenImage21TurboModularPipeline(RBLNModularPipelineMixin, QwenImage21TurboModularPipeline):
    """
    RBLN-accelerated Qwen-Image 2.1 Turbo modular pipeline (8 steps, no guidance): text-to-image and image editing.

    The Qwen3-VL prompt encoder (with its vision tower), the transformer and the VAE run on RBLN NPUs, compiled for
    one profile of image sizes (see [`RBLNQwenImage21TurboModularPipelineConfig`]). The diffusers blocks run as they
    are, with the steps that call the prompt encoder and the one that draws the initial noise replaced (see
    `rbln_turbo_blocks`).

    Condition images (`image`) are center-cropped to the closest compiled aspect ratio and resized to its 1024-class
    size, which is what the diffusers pipeline resizes them to. The generated image takes the compiled size (by default
    1024x1024 or 2048x2048); with several compiled sizes `height` and `width` choose one of them.

    Example:
        ```python
        from optimum.rbln import RBLNQwenImage21TurboModularPipeline

        pipe = RBLNQwenImage21TurboModularPipeline.from_pretrained(
            "Qwen/Qwen-Image-2.1-Turbo", export=True, model_save_dir="qwen-image-2.1-turbo-rbln",
            rbln_config={"profile": "1024"},
        )
        image = pipe(prompt="A red fox in the snow", width=1024, height=1024, output="images")[0]
        ```
    """

    original_class = QwenImage21TurboModularPipeline
    _rbln_config_class = RBLNQwenImage21TurboModularPipelineConfig
    _submodules = ["transformer", "text_encoder", "vae"]

    def __init__(self, blocks: Any = None, **kwargs: Any) -> None:
        # The blocks are the diffusers ones with the RBLN steps (see `rbln_turbo_blocks`), whatever a saved pipeline
        # names as its blocks class.
        super().__init__(blocks=rbln_turbo_blocks() if blocks is None else blocks, **kwargs)

    def _fit_condition_image(self, image: PIL.Image.Image | np.ndarray) -> PIL.Image.Image:
        """Center-crops a condition image to the closest aspect ratio the VAE encoder was compiled for and resizes it
        to that size (the size the diffusers resize step gives an image of that aspect ratio)."""
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)
        ratio = math.log(image.width / image.height)
        height, width = min(
            self.vae.rbln_config.encoder_sample_sizes, key=lambda s: abs(math.log(s[1] / s[0]) - ratio)
        )
        return PIL.ImageOps.fit(image, (width, height), method=PIL.Image.Resampling.LANCZOS)

    def handle_additional_kwargs(self, **kwargs):
        output_resolution = kwargs.get("output_resolution", QWENIMAGE21_CONDITION_RESOLUTION)
        if output_resolution != QWENIMAGE21_CONDITION_RESOLUTION:
            raise NotImplementedError(
                f"Condition images are compiled at the {QWENIMAGE21_CONDITION_RESOLUTION} class only "
                f"(`output_resolution={QWENIMAGE21_CONDITION_RESOLUTION}`); pass `height`/`width` for a larger output."
            )
        image = kwargs.get("image")
        if image is not None:
            image = [self._fit_condition_image(img) for img in (image if isinstance(image, list) else [image])]
            kwargs["image"] = image
        # The size the diffusers pipeline generates at, which the VAE must have been compiled for.
        height, width = kwargs.get("height"), kwargs.get("width")
        if (height is None or width is None) and len(self.vae.rbln_config.sample_sizes) == 1:
            height, width = (
                height or self.vae.rbln_config.sample_sizes[0][0],
                width or self.vae.rbln_config.sample_sizes[0][1],
            )
        if height is None or width is None:
            default = (
                qwenimage21_size(output_resolution, image[-1].width, image[-1].height)
                if image is not None
                else (output_resolution, output_resolution)
            )
            height, width = height or default[0], width or default[1]
        size = (height // 32 * 32, width // 32 * 32)
        if size not in self.vae.rbln_config.sample_sizes:
            raise ValueError(
                f"The pipeline was not compiled for {size[0]}x{size[1]} images ({self.vae.rbln_config.sample_sizes}); "
                "pass `height`/`width` of one of them."
            )
        kwargs["height"], kwargs["width"] = size
        return super().handle_additional_kwargs(**kwargs)


__all__ = [
    "RBLNQwenImage21TurboModularPipeline",
]
