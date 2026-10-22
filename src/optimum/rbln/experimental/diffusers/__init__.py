# Copyright 2026 Rebellions Inc. All rights reserved.

from typing import TYPE_CHECKING

from transformers.utils import _LazyModule

from ...utils.import_utils import define_import_structure


if TYPE_CHECKING:
    from .configurations import *
    from .modeling_modular_diffusers import *
    from .models import *
    from .pipelines import *
else:
    import sys

    _file = globals()["__file__"]
    sys.modules[__name__] = _LazyModule(__name__, _file, define_import_structure(_file), module_spec=__spec__)
