"""Finite-volume layer (v2).

The wide-domain harmonic Zeta kernel, the general box matrices and the JLS
assembly layer live here.
"""

from __future__ import annotations

from .box import mixed_box, orbital_box
from .coupled_s import (
    cm_quantization_matrix,
    coupled_s_roots,
    phase_space_inverse,
    scalar_box,
    simple_quantization_matrix,
)
from .jls_matrix import (
    WEIGHTING_SCALE,
    WEIGHTING_THRESHOLD,
    JMatrixModel,
    RootScanError,
    assemble_inverse,
    quantization_roots,
    row_matrix,
)
from .jls_channels import ChannelLayout, normalise_channel_layout, normalize_channel_layout
from .root_search import matrix_root_residual
from .threshold import threshold_factor, threshold_scaled_inverse, two_k
from .zeta import (
    DEFAULT_POLE_TOLERANCE,
    DEFAULT_TOL,
    MAX_D2,
    MAX_ELL,
    POLE_POLICIES,
    Q2_MAX,
    Q2_MIN,
    FreePoleError,
    ZetaResult,
    harmonic_zeta_wide,
)

__all__ = [
    "matrix_root_residual",
    "JMatrixModel",
    "RootScanError",
    "WEIGHTING_SCALE",
    "WEIGHTING_THRESHOLD",
    "assemble_inverse",
    "ChannelLayout",
    "cm_quantization_matrix",
    "coupled_s_roots",
    "mixed_box",
    "orbital_box",
    "phase_space_inverse",
    "quantization_roots",
    "normalise_channel_layout",
    "normalize_channel_layout",
    "row_matrix",
    "scalar_box",
    "simple_quantization_matrix",
    "threshold_factor",
    "threshold_scaled_inverse",
    "two_k",
    "DEFAULT_POLE_TOLERANCE",
    "DEFAULT_TOL",
    "MAX_D2",
    "MAX_ELL",
    "POLE_POLICIES",
    "Q2_MAX",
    "Q2_MIN",
    "FreePoleError",
    "ZetaResult",
    "harmonic_zeta_wide",
]
