# Copyright 2025 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from diffusers import Flux2KleinPipeline
import os

from ....utils.logging import get_logger
from ...configurations import RBLNFlux2KleinPipelineConfig
from ...modeling_diffusers import RBLNDiffusionMixin


logger = get_logger(__name__)


class RBLNFlux2KleinPipeline(RBLNDiffusionMixin, Flux2KleinPipeline):
    original_class = Flux2KleinPipeline
    _rbln_config_class = RBLNFlux2KleinPipelineConfig
    _submodules = ["text_encoder", "transformer", "vae"]
    use_diffusers_transformer = os.getenv("USE_DIFFUSERS_TRANSFORMER", "0") == "1"
    if use_diffusers_transformer:
        _submodules = ["text_encoder", "vae"]

    def handle_additional_kwargs(self, **kwargs):
        if "max_sequence_length" in kwargs and kwargs["max_sequence_length"] != self.text_encoder.rbln_config.max_seq_len:
            logger.warning(
                f"The text_encoder in this pipeline is compiled with 'max_sequence_length={self.text_encoder.rbln_config.max_seq_len}'. "
                "'max_sequence_length' set by the user will be ignored."
            )
            kwargs["max_sequence_length"] = self.text_encoder.rbln_config.max_seq_len

        compiled_image_size = self.get_compiled_image_size()
        if compiled_image_size is not None:
            if "height" in kwargs and kwargs["height"] != compiled_image_size[0]:
                logger.warning(
                    f"The VAE in this pipeline is compiled with 'height={compiled_image_size[0]}'. "
                    "'height' set by the user will be ignored."
                )
            if "width" in kwargs and kwargs["width"] != compiled_image_size[1]:
                logger.warning(
                    f"The VAE in this pipeline is compiled with 'width={compiled_image_size[1]}'. "
                    "'width' set by the user will be ignored."
                )
            kwargs["height"] = compiled_image_size[0]
            kwargs["width"] = compiled_image_size[1]

        return kwargs
