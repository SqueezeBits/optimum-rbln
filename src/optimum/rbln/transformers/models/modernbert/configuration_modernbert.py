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

from ...configuration_generic import RBLNModelForMaskedLMConfig, RBLNTransformerEncoderForFeatureExtractionConfig


class RBLNModernBertModelConfig(RBLNTransformerEncoderForFeatureExtractionConfig):
    """
    Configuration class for RBLNModernBertModel.

    This configuration class stores the configuration parameters specific to
    RBLN-optimized ModernBERT encoders that return the last hidden state, for feature
    extraction and for task heads that run outside the encoder.
    """


class RBLNModernBertForMaskedLMConfig(RBLNModelForMaskedLMConfig):
    """
    Configuration class for RBLNModernBertForMaskedLM.

    This configuration class stores the configuration parameters specific to
    RBLN-optimized ModernBERT models for masked language modeling tasks.
    """


__all__ = [
    "RBLNModernBertForMaskedLMConfig",
    "RBLNModernBertModelConfig",
]
