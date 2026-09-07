"""
Deterministic physical lighting perturbations for FieldSight-Lite.
Provides 5 realistic lighting stress transformations: Shadow Ramp, Specular Glare,
Overexposure, Underexposure, and Non-Linear Gamma Shifts.
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import cv2
import numpy as np


def apply_shadow_ramp(image_bgr: np.ndarray, min_scale: float = 0.35) -> np.ndarray:
    """
    Applies a smooth diagonal shadow gradient mimicking outdoor cloud or canopy shadows.
    """
    h, w = image_bgr.shape[:2]
    # Create diagonal gradient from top-left (1.0) to bottom-right (min_scale)
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    diag = (x_coords / max(w - 1, 1) + y_coords / max(h - 1, 1)) / 2.0
    gradient = 1.0 - (1.0 - min_scale) * diag
    gradient_3ch = np.dstack([gradient] * 3)

    perturbed = (image_bgr.astype(np.float32) * gradient_3ch)
    return np.clip(perturbed, 0, 255).astype(np.uint8)


def apply_specular_glare(
    image_bgr: np.ndarray,
    center_rel: Tuple[float, float] = (0.5, 0.5),
    radius_rel: float = 0.25,
    intensity: float = 180.0
) -> np.ndarray:
    """
    Simulates direct sun glint with localized desaturation and high-luminance hotspot.
    """
    h, w = image_bgr.shape[:2]
    cx = int(center_rel[0] * w)
    cy = int(center_rel[1] * h)
    sigma = max(1.0, float(radius_rel * min(h, w)))

    y_coords, x_coords = np.mgrid[0:h, 0:w]
    dist_sq = (x_coords - cx) ** 2 + (y_coords - cy) ** 2
    glare_kernel = np.exp(-dist_sq / (2.0 * sigma ** 2))

    # Additive white glare in BGR
    glare_bgr = np.dstack([glare_kernel * intensity] * 3)
    perturbed = image_bgr.astype(np.float32) + glare_bgr
    return np.clip(perturbed, 0, 255).astype(np.uint8)


def apply_overexposure(image_bgr: np.ndarray, bias: float = 55.0) -> np.ndarray:
    """
    Simulates camera overexposure (excessive gain/shutter speed).
    """
    perturbed = image_bgr.astype(np.float32) + float(bias)
    return np.clip(perturbed, 0, 255).astype(np.uint8)


def apply_underexposure(image_bgr: np.ndarray, scale: float = 0.50) -> np.ndarray:
    """
    Simulates camera underexposure (low light / shaded canopy).
    """
    perturbed = image_bgr.astype(np.float32) * float(scale)
    return np.clip(perturbed, 0, 255).astype(np.uint8)


def apply_gamma_shift(image_bgr: np.ndarray, gamma: float = 1.8) -> np.ndarray:
    """
    Applies non-linear radiometric power-law transformation (gamma curve).
    """
    inv_gamma = 1.0 / max(gamma, 1e-4)
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(image_bgr, table)


def generate_all_perturbations(image_bgr: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Generates the standard 5-perturbation lighting stress suite for an input image.
    """
    return {
        "Shadow Ramp": apply_shadow_ramp(image_bgr, min_scale=0.35),
        "Specular Glare": apply_specular_glare(image_bgr, center_rel=(0.45, 0.45), radius_rel=0.25, intensity=160.0),
        "Overexposure (+55)": apply_overexposure(image_bgr, bias=55.0),
        "Underexposure (0.5x)": apply_underexposure(image_bgr, scale=0.50),
        "Gamma Shift (γ=1.8)": apply_gamma_shift(image_bgr, gamma=1.8),
    }
