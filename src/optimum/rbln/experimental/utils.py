# Copyright 2026 Rebellions Inc. All rights reserved.

import functools
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import rebel

from ..configuration_utils import RBLNModelConfig
from ..utils.logging import get_logger


logger = get_logger(__name__)

_warned: set[str] = set()


def experimental(cls):
    """Marks a model or pipeline class of `optimum.rbln.experimental`: its first `from_pretrained` warns, once per
    process, that the class is not a supported one."""
    original = cls.from_pretrained.__func__

    @functools.wraps(original)
    def from_pretrained(klass, *args, **kwargs):
        if not _warned:
            _warned.add(klass.__name__)
            logger.warning(
                f"{klass.__name__} is experimental (optimum.rbln.experimental): it is being ported and is not a "
                "supported model yet. Its interface, options and compiled artifacts may change without notice."
            )
        return original(klass, *args, **kwargs)

    cls.from_pretrained = classmethod(from_pretrained)
    return cls


class LazyRuntime:
    """A graph whose runtime is created on first use and can be released, in place of a `rebel.Runtime` for models
    whose graphs do not all fit the device at once (see `RBLNAutoencoderKLQwenImage21`). It takes the arguments of
    `rebel.Runtime` and is called like one.

    A second runtime created from an `RBLNCompiledModel` whose first runtime was destroyed crashes (segmentation fault,
    rebel-compiler 0.12.1.dev163), so once `path` names the saved graph, runtimes after the first are created from it.
    """

    def __init__(self, compiled_model: rebel.RBLNCompiledModel, tensor_type: str = "pt", **runtime_kwargs: Any):
        self.compiled_model: rebel.RBLNCompiledModel | None = compiled_model
        self.runtime_kwargs = {"tensor_type": tensor_type, **runtime_kwargs}
        self.runtime: rebel.Runtime | None = None
        self.path: Path | None = None

    def open(self) -> rebel.Runtime:
        if self.runtime is None:
            if self.compiled_model is None and self.path is None:
                raise RuntimeError("A released graph can only be loaded again from a saved model.")
            source = self.compiled_model if self.compiled_model is not None else str(self.path)
            self.runtime = rebel.Runtime(source, **self.runtime_kwargs)
            if self.path is not None:
                self.compiled_model = None
        return self.runtime

    def close(self) -> None:
        self.runtime = None

    def __call__(self, *args: Any) -> Any:
        return self.open().run(*args)


def runtimes_by_name(
    cls: Any,
    compiled_models: Sequence[rebel.RBLNCompiledModel],
    rbln_config: RBLNModelConfig,
    runtime_cls: Any = rebel.Runtime,
) -> list[Any]:
    """`_create_runtimes` for a model of several graphs: one runtime per compile config, in order, on the device
    `device_map` gives its name (`RBLNModel._create_runtimes` places every graph on the device of the single default
    name)."""
    names = [cfg.compiled_model_name for cfg in rbln_config.compile_cfgs]
    if any(name not in rbln_config.device_map for name in names):
        cls._raise_missing_compiled_file_error(names)
    return [
        runtime_cls(
            compiled_model,
            tensor_type="pt",
            device=rbln_config.device_map[name],
            activate_profiler=rbln_config.activate_profiler,
            timeout=rbln_config.timeout,
        )
        for name, compiled_model in zip(names, compiled_models, strict=True)
    ]


def graphs_by_name(model: Any, rbln_config: RBLNModelConfig) -> dict[str, Any]:
    """The runtimes of a model of several graphs by graph name, in the order of its compile configs. Without runtimes
    (`create_runtimes=False`) `model` is an `UnavailableRuntime`, which indexing returns for every name."""
    return {cfg.compiled_model_name: model[i] for i, cfg in enumerate(rbln_config.compile_cfgs)}
