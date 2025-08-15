"""
Adaptiq Cloud - HTTP client and API management for Adaptiq services
"""

from .http_client import HTTPClient
from .adaptiq_client import AdaptiqClient

__all__ = [
    'HTTPClient',
    'AdaptiqClient'
]

# Version info
__version__ = "0.12.2"  