"""Custom exceptions used across the language-modeling port.

Ported from ``olmo.exceptions``; retains the class hierarchy expected by the
downstream / evaluator modules. The public-facing names keep the ``OLMo``
prefix so imports in the ported ``eval/`` tree don't need to change.
"""

from __future__ import annotations

__all__ = [
    "OLMoError",
    "OLMoConfigurationError",
    "OLMoCliError",
    "OLMoEnvironmentError",
    "OLMoNetworkError",
    "OLMoCheckpointError",
    "OLMoThreadError",
]


class OLMoError(Exception):
    """Base class for all custom OLMo-derived exceptions."""


class OLMoConfigurationError(OLMoError):
    """An error with a configuration file."""


class OLMoCliError(OLMoError):
    """An error from incorrect CLI usage."""


class OLMoEnvironmentError(OLMoError):
    """An error from incorrect environment variables."""


class OLMoNetworkError(OLMoError):
    """An error with a network request."""


class OLMoCheckpointError(OLMoError):
    """An error occurred reading or writing from a checkpoint."""


class OLMoThreadError(Exception):
    """Raised when a thread fails."""
