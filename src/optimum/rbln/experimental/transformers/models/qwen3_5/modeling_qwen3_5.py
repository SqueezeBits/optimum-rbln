# Copyright 2026 Rebellions Inc. All rights reserved.

import os
from tempfile import TemporaryDirectory
from typing import Any

import rebel
from transformers import GenerationConfig, PretrainedConfig, Qwen3_5ForConditionalGeneration

from .....configuration_utils import RBLNModelConfig
from .....modeling_base import RBLNBaseModel
from .....transformers import RBLNQwen3_5ForConditionalGeneration
from .....utils.runtime_utils import UnavailableRuntime
from ....utils import experimental
from .configuration_qwen3_5 import RBLNFlux3PromptUpsamplerConfig


@experimental
class RBLNFlux3PromptUpsampler(RBLNQwen3_5ForConditionalGeneration):
    """
    The Qwen3.5 prompt upsampler of FLUX.3 pipelines: RBLNQwen3_5ForConditionalGeneration as a pipeline component,
    whose generation settings are read from its subfolder of the pipeline directory.
    """

    _hf_class = Qwen3_5ForConditionalGeneration
    _rbln_config_class = RBLNFlux3PromptUpsamplerConfig

    def __init__(
        self,
        models: list[rebel.Runtime] | UnavailableRuntime,
        config: PretrainedConfig,
        rbln_config: RBLNModelConfig,
        model_save_dir: str | os.PathLike[str] | TemporaryDirectory | None = None,
        subfolder: str = "",
        rbln_compiled_models: list[rebel.RBLNCompiledModel] | None = None,
        rbln_submodules: list[RBLNBaseModel] | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            models,
            config,
            rbln_config,
            model_save_dir=model_save_dir,
            subfolder=subfolder,
            rbln_compiled_models=rbln_compiled_models,
            rbln_submodules=rbln_submodules,
            **kwargs,
        )
        # RBLNBaseModel reads `generation_config.json` from the directory itself; a pipeline component's is in its
        # subfolder.
        gen_config_dir = model_save_dir.name if isinstance(model_save_dir, TemporaryDirectory) else model_save_dir
        if subfolder and gen_config_dir is not None:
            try:
                self.generation_config = GenerationConfig.from_pretrained(
                    gen_config_dir, subfolder=subfolder, trust_remote_code=True
                )
                self.generation_config.use_cache = True
            except OSError:
                pass

    @classmethod
    def update_rbln_config_using_pipe(cls, pipe: Any, rbln_config: Any, submodule_name: str) -> Any:
        # The pipeline config sets the prompt length and the attention options.
        return rbln_config


__all__ = [
    "RBLNFlux3PromptUpsampler",
]
