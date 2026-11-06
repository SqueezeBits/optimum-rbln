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

from diffusers.modular_pipelines.flux3.before_denoise import Flux3PrepareLatentsStep
from diffusers.modular_pipelines.flux3.encoders import Flux3TextEncoderStep
from diffusers.modular_pipelines.flux3.prompt_upsampler import MAX_PIXELS_BY_REFS, MAX_REFERENCES, get_reference_size

from ....configuration_utils import RBLNModelConfig
from ....transformers import RBLNQwen3_5ForConditionalGenerationConfig, RBLNQwen3VLTextEncoderConfig
from ..models import RBLNAutoencoderKLFlux2Config, RBLNFlux3Transformer2DModelConfig


# Width:height ratios of the reference-image buckets.
FLUX3_REFERENCE_ASPECT_RATIOS = [(1, 1), (4, 3), (3, 4), (16, 9), (9, 16)]
# References up to this count share the smaller image-conditioned transformer graph.
FLUX3_FEW_REFERENCES = 4
# Text lengths below `max_seq_len` the transformer also gets graphs for: a caption from the prompt upsampler takes
# about 250-330 tokens, one with a layout of a few boxes about 500.
FLUX3_TEXT_TOKEN_BUCKETS = [320, 640]
# What the submodules are compiled for that follows the diffusers FLUX.3 pipeline (checked against the pipeline before
# compiling, see `check_flux3_pipeline`): the text window of the text encoder step, the encoder layers it reads, the
# latent patch (VAE scale factor x 2), the image rotary scale of the latents step, and the vision patch of the prompt
# upsampler.
FLUX3_TEXT_PAD_MULTIPLE = 80
FLUX3_TEXT_ENCODER_OUT_LAYERS = [4, 8, 12, 16, 20, 24, 28, 32]
FLUX3_LATENT_PATCH = 16
FLUX3_IMAGE_ROPE_SCALE = 0.5
FLUX3_UPSAMPLER_VISION_PATCH = 16


def flux3_reference_sizes(num_references: int, aspect_ratios: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """The (height, width) buckets of a reference image when `num_references` are given: one per aspect ratio, at the
    size the FLUX.3 prompt upsampler resizes such a reference to (its pixel budget shrinks with the count)."""
    sizes = set()
    for ratio_w, ratio_h in aspect_ratios:
        width, height = get_reference_size(ratio_w * 1024, ratio_h * 1024, num_references)
        sizes.add((height, width))
    return sorted(sizes)


def _reference_tokens(max_references: int, aspect_ratios: list[tuple[int, int]], patch_size: int) -> int:
    # The most reference tokens up to `max_references` references can carry into the transformer.
    return max(
        n * max(h * w for h, w in flux3_reference_sizes(n, aspect_ratios)) // patch_size**2
        for n in range(1, max_references + 1)
    )


def _find_block(blocks, block_class):
    # The first block of `block_class` in a (nested) modular pipeline block.
    if isinstance(blocks, block_class):
        return blocks
    for block in getattr(blocks, "sub_blocks", {}).values():
        if (found := _find_block(block, block_class)) is not None:
            return found
    return None


def _round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


def check_flux3_pipeline(pipe: Any) -> None:
    """Checks that a diffusers FLUX.3 modular pipeline uses the values the submodules are compiled for.

    Args:
        pipe (ModularPipeline): The FLUX.3 pipeline about to be compiled.

    Raises:
        ValueError: If one of the `FLUX3_*` values differs from the pipeline's.
    """
    text_step = _find_block(pipe._blocks, Flux3TextEncoderStep)
    latents_step = _find_block(pipe._blocks, Flux3PrepareLatentsStep)
    out_layers = next(param.default for param in text_step.inputs if param.name == "text_encoder_out_layers")
    found = {
        "FLUX3_TEXT_PAD_MULTIPLE": (text_step.pad_multiple, FLUX3_TEXT_PAD_MULTIPLE),
        "FLUX3_TEXT_ENCODER_OUT_LAYERS": (list(out_layers), FLUX3_TEXT_ENCODER_OUT_LAYERS),
        "FLUX3_LATENT_PATCH": (pipe.vae_scale_factor * 2, FLUX3_LATENT_PATCH),
        "FLUX3_IMAGE_ROPE_SCALE": (latents_step.image_rope_scale, FLUX3_IMAGE_ROPE_SCALE),
        "FLUX3_UPSAMPLER_VISION_PATCH": (
            pipe.prompt_upsampler.config.vision_config.patch_size,
            FLUX3_UPSAMPLER_VISION_PATCH,
        ),
    }
    mismatches = [f"{name}: pipeline {got}, compiled for {want}" for name, (got, want) in found.items() if got != want]
    if mismatches:
        raise ValueError("The FLUX.3 pipeline differs from what RBLN compiles it for: " + "; ".join(mismatches))


class RBLNFlux3ModularPipelineConfig(RBLNModelConfig):
    """
    Configuration class for the RBLN FLUX.3 modular pipeline (FLUX 3 Image Klein base, with classifier-free guidance).

    Classifier-free guidance runs the transformer once for the prompt and once for the negative prompt, each with a
    batch of one, so the submodules are compiled exactly as for the step-distilled pipeline.

    The image size and the maximum prompt length are fixed at compile time and shared by the submodules: the text
    encoder and the transformer are compiled for the same `max_seq_len`, and the transformer and the VAE for the same
    image size. The prompt upsampler, which writes the caption the text encoder reads, has its own length.

    The whole pipeline fits one RBLN-CR13 device. The VAE graphs keep most of their memory on the first chiplet of
    the device; to spread it, place the VAE or the transformer on another device, e.g.
    `rbln_config={"vae": {"device": 1}}`. The supported image sizes are those of the 1024 class (about one megapixel),
    which fit one device; larger sizes are not supported.
    """

    submodules = ["prompt_upsampler", "text_encoder", "transformer", "vae"]

    def __init__(
        self,
        text_encoder: RBLNQwen3VLTextEncoderConfig | None = None,
        transformer: RBLNFlux3Transformer2DModelConfig | None = None,
        vae: RBLNAutoencoderKLFlux2Config | None = None,
        prompt_upsampler: RBLNQwen3_5ForConditionalGenerationConfig | None = None,
        *,
        max_references: int | None = None,
        reference_aspect_ratios: list[tuple[int, int]] | None = None,
        height: int | None = None,
        width: int | None = None,
        max_seq_len: int | None = None,
        text_token_buckets: list[int] | None = None,
        batch_size: int | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            text_encoder (RBLNQwen3VLTextEncoderConfig | None): Configuration for the Qwen3-VL text encoder.
                Initialized as RBLNQwen3VLTextEncoderConfig with the RBLN-CR13 precision option (`exp_silu`) if not
                provided.
            transformer (RBLNFlux3Transformer2DModelConfig | None): Configuration for the transformer.
                Initialized as RBLNFlux3Transformer2DModelConfig if not provided.
            vae (RBLNAutoencoderKLFlux2Config | None): Configuration for the VAE: the decoder at the image size and
                the encoder at the reference sizes.
                Initialized as RBLNAutoencoderKLFlux2Config if not provided.
            prompt_upsampler (RBLNQwen3_5ForConditionalGenerationConfig | None): Configuration for the prompt
                upsampler. Initialized as RBLNQwen3_5ForConditionalGenerationConfig if not provided, with room for
                the system prompt, the reference images and the step's 3584 new tokens (`max_seq_len` 8192, flash
                attention with `kvcache_partition_len` 4096) and a vision encoder for one reference at its largest
                size.
            max_references (int | None): The most reference images a call can take, from 0 (text-to-image only) to 10.
                Defaults to 10.
            reference_aspect_ratios (list[tuple[int, int]] | None): Width:height ratios of the reference buckets. A
                reference image is center-cropped to the closest ratio and resized to the size the prompt upsampler
                uses for it, which shrinks with the number of references (1024x1024 pixels for one, 724x724 for two,
                512x512 for three or more); the VAE encoder is compiled for each of these sizes. Defaults to 1:1, 4:3,
                3:4, 16:9 and 9:16.
            height (int | None): Height of the generated images, a multiple of the latent patch (16 pixels for FLUX.3).
                Defaults to 1024.
            width (int | None): Width of the generated images, a multiple of the latent patch. Defaults to 1024.
            max_seq_len (int | None): Maximum prompt length in tokens (after the chat template), for the text
                encoder and the transformer. Rounded up to the text window of the pipeline (80 tokens for FLUX.3).
                Defaults to 1040 rather than the 8192 of the diffusers pipeline, since the text tokens join every
                transformer step: captions from the prompt upsampler take about 250-600 tokens, but a layout adds
                about 80 per box (about 950 tokens with 8 boxes, 1700 with 20). Longer prompts are truncated, with a
                warning; raise it (e.g. to 2080) for layouts with many boxes.
            text_token_buckets (list[int] | None): Shorter text lengths the transformer also gets graphs for, so that
                a prompt runs the smallest graph that holds it instead of being padded to `max_seq_len` (a step is
                about a quarter faster at 320 than at 1040 tokens). Rounded up to the text window. Defaults to 320
                and 640.
            batch_size (int | None): Batch size for inference, applied to all submodules. Defaults to 1.
            kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If only one of height and width is given, if they are not multiples of the latent patch,
                or if max_references is out of range.
        """
        super().__init__(**kwargs)
        if (height is None) != (width is None):
            raise ValueError("Both height and width must be provided together if used")
        height, width = height or 1024, width or 1024
        max_references = MAX_REFERENCES if max_references is None else max_references
        if not 0 <= max_references <= MAX_REFERENCES:
            raise ValueError(f"max_references must be in [0, {MAX_REFERENCES}], got {max_references}")
        aspect_ratios = [(ratio[0], ratio[1]) for ratio in reference_aspect_ratios or FLUX3_REFERENCE_ASPECT_RATIOS]
        counts = range(1, max_references + 1)
        encoder_sample_sizes = sorted({size for n in counts for size in flux3_reference_sizes(n, aspect_ratios)})

        if height % FLUX3_LATENT_PATCH or width % FLUX3_LATENT_PATCH:
            raise ValueError(f"height and width must be multiples of {FLUX3_LATENT_PATCH}, got {height}x{width}")
        max_seq_len = _round_up(max_seq_len or 1040, FLUX3_TEXT_PAD_MULTIPLE)
        text_token_buckets = FLUX3_TEXT_TOKEN_BUCKETS if text_token_buckets is None else text_token_buckets
        text_buckets = sorted({_round_up(n, FLUX3_TEXT_PAD_MULTIPLE) for n in text_token_buckets} | {max_seq_len})
        capacity_counts = {min(FLUX3_FEW_REFERENCES, max_references), max_references} - {0}
        reference_token_buckets = sorted(
            {_reference_tokens(n, aspect_ratios, FLUX3_LATENT_PATCH) for n in capacity_counts}
        )
        # The vision encoder of the upsampler takes one reference at its largest size.
        visual = {"max_seq_len": MAX_PIXELS_BY_REFS[1] // FLUX3_UPSAMPLER_VISION_PATCH**2}
        if isinstance(prompt_upsampler, dict) and isinstance(prompt_upsampler.get("visual"), dict):
            prompt_upsampler = {**prompt_upsampler, "visual": {**visual, **prompt_upsampler["visual"]}}

        self.text_encoder = self.initialize_submodule_config(
            text_encoder,
            cls_name="RBLNQwen3VLTextEncoderConfig",
            batch_size=batch_size,
            exp_silu=True,
            max_seq_len=max_seq_len,
            hidden_states_layers=FLUX3_TEXT_ENCODER_OUT_LAYERS,
        )
        self.transformer = self.initialize_submodule_config(
            transformer,
            cls_name="RBLNFlux3Transformer2DModelConfig",
            batch_size=batch_size,
            sample_size=(height // FLUX3_LATENT_PATCH, width // FLUX3_LATENT_PATCH),
            max_seq_len=max_seq_len,
            text_token_buckets=[n for n in text_buckets if n <= max_seq_len],
            image_rope_scale=FLUX3_IMAGE_ROPE_SCALE,
            reference_token_buckets=reference_token_buckets,
        )
        self.prompt_upsampler = self.initialize_submodule_config(
            prompt_upsampler,
            cls_name="RBLNQwen3_5ForConditionalGenerationConfig",
            batch_size=1,
            max_seq_len=8192,
            # Flash attention, as the Qwen3.5 tests use.
            attn_impl="flash_attn",
            kvcache_partition_len=4096,
            visual=visual,
        )
        self.vae = self.initialize_submodule_config(
            vae,
            cls_name="RBLNAutoencoderKLFlux2Config",
            batch_size=batch_size,
            sample_size=(height, width),
            encoder_sample_size=encoder_sample_sizes or None,
        )

    @property
    def image_size(self) -> tuple[int, int]:
        return self.vae.sample_size


class RBLNFlux3DistilledModularPipelineConfig(RBLNFlux3ModularPipelineConfig):
    """
    Configuration class for the RBLN step-distilled FLUX.3 modular pipeline (FLUX 3 Image Klein).

    It takes the options of [`RBLNFlux3ModularPipelineConfig`].
    """


__all__ = [
    "RBLNFlux3DistilledModularPipelineConfig",
    "RBLNFlux3ModularPipelineConfig",
]
