# SPDX-License-Identifier: LGPL-3.0-or-later
"""Backend-agnostic interfaces for long-range energy plugins.

This module contains the plugin and interaction-kernel registries and the
non-periodic direct latent-charge implementation. The existing specialized
long-range models retain their independent legacy paths.
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
from .latent_charge import (
    LatentChargeLongRange,
)
from .latent_multipole import (
    LatentMultipoleLongRange,
)
from .migration import (
    migrate_dpa4c_lr_config,
    migrate_dpa4c_lr_record,
)

__all__ = [
    "COULOMB_CONSTANT",
    "BaseInteractionKernel",
    "BaseLongRange",
    "LESKernel",
    "LatentChargeLongRange",
    "LatentMultipoleLongRange",
    "SOGKernel",
    "check_long_range_support",
    "make_base_long_range",
    "migrate_dpa4c_lr_config",
    "migrate_dpa4c_lr_record",
]
