# src/adaptiq/__init__.py

__version__ = "0.12.2"  

try:
    from .abstract import *
    from .pipelines import *
    from .q_table import *
    from .reporting import *
    from .entities import *
    
except ImportError:
    pass


def get_version():
    return __version__


import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
