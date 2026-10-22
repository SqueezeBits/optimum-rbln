# Copyright 2026 Rebellions Inc. All rights reserved.

from typing import TYPE_CHECKING

from transformers.utils import _LazyModule

from .....utils.import_utils import define_import_structure


if TYPE_CHECKING:
    from .configuration_autoencoder_kl_flux2 import *
    from .configuration_autoencoder_kl_qwenimage21 import *
    from .configuration_transformer_flux3 import *
    from .configuration_transformer_qwenimage21 import *
else:
    import sys

    _file = globals()["__file__"]
    sys.modules[__name__] = _LazyModule(__name__, _file, define_import_structure(_file), module_spec=__spec__)
