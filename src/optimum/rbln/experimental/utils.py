# Copyright 2026 Rebellions Inc. All rights reserved.

import functools

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
