"""
Pre-flight image quality assessment engine for FieldSight-Lite.
Evaluates physical CIELAB lightness, specular glare ratio, deep shadow ratio, and blur.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np

from src.schemas import QualityMetrics, QualityStatus


class ImageQualityAssessor:
    """
    Pre-flight quality analyzer ensuring input imagery meets optical clarity constraints
    prior to colorimetric segmentation. Prevents corrupted inference under extreme conditions.
    """

    def __init__(
        self,
        min_laplacian_var: float = 100.0,
        max_glare_ratio: float = 0.15,
        max_shadow_ratio: float = 0.35,
        glare_l_physical_thresh: float = 96.0,
        shadow_l_physical_thresh: float = 10.0,
        glare_sat_max: int = 30
    ):
        self.min_laplacian_var = float(min_laplacian_var)
        self.max_glare_ratio = float(max_glare_ratio)
        self.max_shadow_ratio = float(max_shadow_ratio)
        self.glare_l_physical_thresh = float(glare_l_physical_thresh)
        self.shadow_l_physical_thresh = float(shadow_l_physical_thresh)
        self.glare_sat_max = int(glare_sat_max)

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> ImageQualityAssessor:
        q_cfg = cfg.get("quality", {})
        return cls(
            min_laplacian_var=q_cfg.get("min_laplacian_var", 100.0),
            max_glare_ratio=q_cfg.get("max_glare_ratio", 0.15),
            max_shadow_ratio=q_cfg.get("max_shadow_ratio", 0.35),
            glare_l_physical_thresh=q_cfg.get("glare_l_physical_thresh", 96.0),
            shadow_l_physical_thresh=q_cfg.get("shadow_l_physical_thresh", 10.0),
            glare_sat_max=q_cfg.get("glare_sat_max", 30)
        )

    def assess_quality(self, bgr_image: np.ndarray) -> QualityMetrics:
        """
        Executes pre-flight optical validation on an input BGR image.
        Returns immutable QualityMetrics.
        """
        if bgr_image is None or not isinstance(bgr_image, np.ndarray):
            return QualityMetrics(
                status=QualityStatus.UNUSABLE,
                mean_lightness=0.0,
                std_lightness=0.0,
                glare_pixel_ratio=0.0,
                shadow_pixel_ratio=0.0,
                laplacian_variance=0.0,
                is_valid_for_processing=False,
                diagnostic_message="Input image is None or invalid array type"
            )

        if len(bgr_image.shape) != 3 or bgr_image.shape[2] != 3:
            return QualityMetrics(
                status=QualityStatus.UNUSABLE,
                mean_lightness=0.0,
                std_lightness=0.0,
                glare_pixel_ratio=0.0,
                shadow_pixel_ratio=0.0,
                laplacian_variance=0.0,
                is_valid_for_processing=False,
                diagnostic_message=f"Expected 3-channel BGR image, got shape {bgr_image.shape}"
            )

        h, w = bgr_image.shape[:2]
        total_pixels = h * w
        if total_pixels == 0:
            return QualityMetrics(
                status=QualityStatus.UNUSABLE,
                mean_lightness=0.0,
                std_lightness=0.0,
                glare_pixel_ratio=0.0,
                shadow_pixel_ratio=0.0,
                laplacian_variance=0.0,
                is_valid_for_processing=False,
                diagnostic_message="Image has zero spatial dimensions"
            )

        # Convert to Grayscale for focus calculation
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # Convert to CIELAB and HSV
        lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

        # OpenCV CIELAB L channel is [0, 255]. Convert to physical CIELAB L* [0, 100]
        L_8bit = lab[:, :, 0].astype(np.float64)
        L_physical = (L_8bit / 255.0) * 100.0
        S_hsv = hsv[:, :, 1]

        mean_l = float(np.mean(L_physical))
        std_l = float(np.std(L_physical))

        # Specular glare: High physical lightness AND low chromatic saturation
        glare_mask = (L_physical >= self.glare_l_physical_thresh) & (S_hsv <= self.glare_sat_max)
        glare_count = int(np.count_nonzero(glare_mask))
        glare_ratio = float(glare_count / total_pixels)

        # Deep shadow: Very low physical lightness
        shadow_mask = (L_physical <= self.shadow_l_physical_thresh)
        shadow_count = int(np.count_nonzero(shadow_mask))
        shadow_ratio = float(shadow_count / total_pixels)

        # Boundary and status classification
        if lap_var < self.min_laplacian_var:
            status = QualityStatus.BLURRY
            is_valid = False
            diag_msg = f"Blurry image: Laplacian variance {lap_var:.1f} < threshold {self.min_laplacian_var:.1f}"
        elif glare_ratio > self.max_glare_ratio:
            status = QualityStatus.DEGRADED_GLARE
            is_valid = False
            diag_msg = f"Excessive glare: Glare ratio {glare_ratio:.2%} > threshold {self.max_glare_ratio:.2%}"
        elif shadow_ratio > self.max_shadow_ratio:
            status = QualityStatus.DEGRADED_SHADOW
            is_valid = False
            diag_msg = f"Excessive shadow: Shadow ratio {shadow_ratio:.2%} > threshold {self.max_shadow_ratio:.2%}"
        elif (glare_ratio > 0.05) or (shadow_ratio > 0.15):
            status = QualityStatus.ACCEPTABLE
            is_valid = True
            diag_msg = f"Acceptable image with moderate illumination variance (glare={glare_ratio:.1%}, shadow={shadow_ratio:.1%})"
        else:
            status = QualityStatus.EXCELLENT
            is_valid = True
            diag_msg = "Excellent optical clarity and balanced exposure"

        return QualityMetrics(
            status=status,
            mean_lightness=mean_l,
            std_lightness=std_l,
            glare_pixel_ratio=glare_ratio,
            shadow_pixel_ratio=shadow_ratio,
            laplacian_variance=lap_var,
            is_valid_for_processing=is_valid,
            diagnostic_message=diag_msg
        )
