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

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2

from ...configurations import RBLNAutoencoderKLFlux2Config
from .autoencoder_kl import RBLNAutoencoderKL


if TYPE_CHECKING:
    from ....configuration_utils import RBLNModelConfig
    from transformers import PreTrainedModel


class RBLNAutoencoderKLFlux2(RBLNAutoencoderKL):
    auto_model_class = AutoencoderKLFlux2
    _rbln_config_class = RBLNAutoencoderKLFlux2Config

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        artifacts_path = self.model_save_dir / self.subfolder / "torch_artifacts.pth"
        if artifacts_path.exists():
            artifacts = torch.load(artifacts_path, weights_only=False)
            bn_state = artifacts["bn"]
            self.bn = torch.nn.BatchNorm2d(
                bn_state["running_mean"].shape[0],
                eps=self.config.batch_norm_eps,
                momentum=self.config.batch_norm_momentum,
                affine=False,
                track_running_stats=True,
            )
            self.bn.load_state_dict(bn_state)

    @classmethod
    def save_torch_artifacts(
        cls,
        model: "PreTrainedModel",
        save_dir_path: Path,
        subfolder: str,
        rbln_config: "RBLNModelConfig",
    ):
        torch.save({"bn": model.bn.state_dict()}, save_dir_path / subfolder / "torch_artifacts.pth")
