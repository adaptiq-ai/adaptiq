# src/adaptiq/core/__init__.py

try:
    from .abstract import *
    from .entities import *
    from .pipelines import *
    from .q_table import *
    from .reporting import *

except ImportError:
    pass


def get_version():
    # Single source of truth: the package version declared in pyproject.toml
    # and mirrored in adaptiq/__init__.py.
    from adaptiq import __version__

    return __version__


import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
