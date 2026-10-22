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

from .....diffusers.configurations.models.configuration_autoencoder_kl import RBLNAutoencoderKLConfig


class RBLNAutoencoderKLFlux2Config(RBLNAutoencoderKLConfig):
    """
    Configuration class for the RBLN FLUX.2 Variational Autoencoder, used by FLUX.2 and FLUX.3 pipelines.

    It takes the options of [`RBLNAutoencoderKLConfig`], plus the image sizes to compile the encoder for: FLUX
    pipelines encode reference images, which need not have the size of the generated image.
    """

    def __init__(
        self,
        encoder_sample_sizes: list[tuple[int, int]] | None = None,
        share_weights: bool | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            encoder_sample_sizes (list[tuple[int, int]] | None): The (height, width) image sizes to compile the encoder
                for, one graph per size; `encode` runs the graph of its input's size. Setting it implies
                `uses_encoder`. Defaults to `[sample_size]` when `uses_encoder` is set.
            share_weights (bool | None): Whether the encoder graphs share the encoder weights on the device instead of
                holding a copy each: the 15 reference sizes of FLUX 3 Image Klein take 6.9 GiB on RBLN-CR13 instead of
                11.3. Defaults to True.
            kwargs: Additional arguments passed to RBLNAutoencoderKLConfig.
        """
        super().__init__(**kwargs)
        if encoder_sample_sizes is not None:
            self.uses_encoder = True
            encoder_sample_sizes = sorted({(size[0], size[1]) for size in encoder_sample_sizes})
        self.encoder_sample_sizes = encoder_sample_sizes
        self.share_weights = True if share_weights is None else share_weights


__all__ = [
    "RBLNAutoencoderKLFlux2Config",
]
