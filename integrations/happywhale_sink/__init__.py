"""HappyWhale API connector package.

Exposes :class:`HappyWhaleConnector` for submitting cetacean observations to
the HappyWhale public API (see https://happywhale.com).
"""

from .connector import HappyWhaleAPIError, HappyWhaleConnector

__all__ = ["HappyWhaleConnector", "HappyWhaleAPIError"]
