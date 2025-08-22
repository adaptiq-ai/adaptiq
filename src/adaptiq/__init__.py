# src/adaptiq/__init__.py

__version__ = "0.12.4"


try:
    from .agents import *
    from .core import *

except ImportError:
    pass

import importlib.resources

with importlib.resources.path(
    "adaptiq.templates.crew_template", "__init__.py"
) as template_path:
    template_source = str(template_path.parent)


def get_version():
    return __version__


import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
