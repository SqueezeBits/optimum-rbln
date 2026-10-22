# Copyright 2026 Rebellions Inc. All rights reserved.

from os import PathLike
from typing import Any

from typing_extensions import Self

from ...configuration_utils import RBLNModelConfig
from ...diffusers.modeling_diffusers import RBLNDiffusionMixin
from ...modeling import RBLNModel


class RBLNModularPipelineMixin(RBLNDiffusionMixin):
    """
    RBLNDiffusionMixin for diffusers `ModularPipeline`s, whose components are loaded and saved through component specs
    (`modular_model_index.json`) instead of `model_index.json`.

    Compilation reuses `RBLNDiffusionMixin`: the submodules listed in `_submodules` are loaded as PyTorch modules,
    compiled one by one with their entry of the pipeline's RBLN config, and swapped into the pipeline with
    `update_components`. The saved pipeline records each compiled submodule as an `optimum.rbln` component spec that
    points at its subfolder, and `from_pretrained` loads it back from there.

    To use this mixin, inherit from it and from the diffusers modular pipeline, and list the compiled submodules:

    Example:
        ```python
        class RBLNFlux3DistilledModularPipeline(RBLNModularPipelineMixin, Flux3DistilledModularPipeline):
            original_class = Flux3DistilledModularPipeline
            _submodules = ["prompt_upsampler", "text_encoder", "transformer", "vae"]
        ```

    Class Variables:
        - `_submodules`: The components compiled for RBLN NPUs.
    """

    @classmethod
    def from_pretrained(  # type: ignore[override]
        cls,
        model_id: str,
        *,
        export: bool | None = None,
        model_save_dir: str | PathLike[str] | None = None,
        rbln_config: dict[str, Any] | RBLNModelConfig | None = None,
        **kwargs: Any,
    ) -> Self:
        """
        Load a modular diffusion pipeline, compiling its submodules for RBLN NPUs when `export=True`.

        Args:
            model_id (`str`):
                The model ID on the Hugging Face Hub or the local path of the pipeline.
            export:
                If True, loads the PyTorch submodules from `model_id` and compiles them. If False, loads already
                compiled submodules from `model_id`. If None, compiles when `model_id` has no compiled submodules.
            model_save_dir:
                Directory to save the compiled pipeline to. Only used when `export=True`.
            rbln_config:
                RBLN configuration of the pipeline, with optional per-submodule entries.
            kwargs:
                Hub arguments (e.g. `token`, `revision`) and `rbln_`-prefixed configuration options.

        Returns:
            The pipeline, with its submodules running on RBLN NPUs.
        """
        rbln_config, kwargs = cls.get_rbln_config_class().initialize_from_kwargs(rbln_config, **kwargs)
        if export is None:
            export = any(
                not RBLNModel._is_compiled(
                    model_id,
                    token=kwargs.get("token"),
                    revision=kwargs.get("revision"),
                    force_download=kwargs.get("force_download", False),
                    cache_dir=kwargs.get("cache_dir"),
                    subfolder=submodule_name,
                    local_files_only=kwargs.get("local_files_only", False),
                )
                for submodule_name in cls._submodules
            )

        pipe = cls(pretrained_model_name_or_path=model_id, **kwargs)  # type: ignore[call-arg]
        other_components = [
            name
            for name, spec in pipe._component_specs.items()
            if spec.default_creation_method == "from_pretrained" and name not in cls._submodules
        ]
        if not export:
            # A compiled pipeline records no paths (see `_save_compiled_pipeline`): its components are in `model_id`.
            for name in other_components:
                if pipe._component_specs[name].pretrained_model_name_or_path is None:
                    pipe._component_specs[name].pretrained_model_name_or_path = str(model_id)
        if other_components:
            pipe.load_components(names=other_components)

        if export:
            pipe.load_components(names=cls._submodules)
            compiled_submodules = cls._compile_submodules(pipe, {}, model_save_dir, rbln_config)
        else:
            compiled_submodules = cls._load_compiled_submodules(pipe, model_id, rbln_config)

        pipe.update_components(**compiled_submodules)
        if export and model_save_dir is not None:
            cls._save_compiled_pipeline(pipe, model_save_dir)
        return pipe

    @classmethod
    def _load_compiled_submodules(
        cls, pipe: Self, model_id: str, rbln_config: RBLNModelConfig
    ) -> dict[str, RBLNModel]:
        # No ContextRblnConfig here: a context would override the submodules' own runtime options (e.g.
        # `transformer={"create_runtimes": False}`), and the submodule configs already inherit the pipeline's.
        submodules = {}
        for submodule_name in cls._submodules:
            submodule_cls = pipe._component_specs[submodule_name].type_hint
            if submodule_cls is None or not issubclass(submodule_cls, RBLNModel):
                raise ValueError(
                    f"The pipeline at '{model_id}' has no compiled '{submodule_name}'. "
                    "If you want to compile, set `export=True`."
                )
            submodule_config = getattr(rbln_config, submodule_name)
            if isinstance(submodule_config, RBLNModelConfig):
                submodule_config = submodule_config.get_load_overrides()
            submodules[submodule_name] = submodule_cls.from_pretrained(
                model_id, export=False, subfolder=submodule_name, rbln_config=submodule_config
            )
        return submodules

    @classmethod
    def _save_compiled_pipeline(cls, pipe: Self, model_save_dir: str | PathLike[str]) -> None:
        from diffusers.modular_pipelines.modular_pipeline_utils import ComponentSpec

        # The compiled submodules were saved in their subfolders while compiling; save the rest around them.
        compiled_submodules = {name: getattr(pipe, name) for name in cls._submodules}
        for name in compiled_submodules:
            setattr(pipe, name, None)
        pipe.save_pretrained(model_save_dir, overwrite_modular_index=True)
        for name, submodule in compiled_submodules.items():
            setattr(pipe, name, submodule)
            spec = ComponentSpec(
                name=name,
                type_hint=type(submodule),
                pretrained_model_name_or_path=str(model_save_dir),
                subfolder=name,
            )
            pipe._component_specs[name] = spec
            pipe.register_to_config(
                **{name: ("optimum.rbln", type(submodule).__name__, pipe._component_spec_to_dict(spec))}
            )
        # Record no paths, so that the directory can be moved or uploaded: `from_pretrained` loads every component
        # from the directory it is given. The compiled components are typed by their name in `optimum.rbln` rather
        # than by their module, so that the artifact outlives a move of the class (e.g. out of `experimental`).
        for name, value in dict(pipe.config).items():
            if isinstance(value, (list, tuple)) and len(value) == 3 and isinstance(value[2], dict):
                library, class_name, spec_dict = value
                spec_dict = {**spec_dict, "pretrained_model_name_or_path": None}
                if library == "optimum.rbln":
                    spec_dict["type_hint"] = (library, class_name)
                pipe.register_to_config(**{name: (library, class_name, spec_dict)})
        pipe.save_config(model_save_dir)

    def handle_additional_kwargs(self, **kwargs):
        # The image size is fixed at compile time; fill it in when the caller leaves it out.
        compiled_image_size = self.get_compiled_image_size()
        if compiled_image_size is not None:
            if kwargs.get("height") is None:
                kwargs["height"] = compiled_image_size[0]
            if kwargs.get("width") is None:
                kwargs["width"] = compiled_image_size[1]
        return kwargs

    def __call__(self, state=None, output=None, **kwargs):
        kwargs = self.handle_additional_kwargs(**kwargs)
        return super(RBLNDiffusionMixin, self).__call__(state, output, **kwargs)  # type: ignore[misc]


__all__ = [
    "RBLNModularPipelineMixin",
]
