# SPDX-License-Identifier: LGPL-3.0-or-later
"""Backend-agnostic interfaces for long-range energy plugins.

This module contains the plugin and interaction-kernel registries. Concrete
latent-charge and latent-multipole implementations are introduced separately;
the existing specialized long-range models do not use this path yet.
"""

from .base_long_range import (
    BaseLongRange,
    check_long_range_support,
    make_base_long_range,
)
from .kernel import (
    COULOMB_CONSTANT,
    BaseInteractionKernel,
    LESKernel,
    SOGKernel,
)

__all__ = [
    "COULOMB_CONSTANT",
    "BaseInteractionKernel",
    "BaseLongRange",
    "LESKernel",
    "SOGKernel",
    "check_long_range_support",
    "make_base_long_range",
]
