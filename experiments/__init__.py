"""
Experimental benchmarking suite and baseline implementations for FieldSight-Lite.
"""

from experiments.synthetic_perturb import (
    apply_shadow_ramp,
    apply_specular_glare,
    apply_overexposure,
    apply_underexposure,
    apply_gamma_shift,
    generate_all_perturbations,
)

__all__ = [
    "apply_shadow_ramp",
    "apply_specular_glare",
    "apply_overexposure",
    "apply_underexposure",
    "apply_gamma_shift",
    "generate_all_perturbations",
]
