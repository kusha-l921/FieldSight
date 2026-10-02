"""
Severity quantification and multi-factor confidence assessment engine for FieldSight-Lite.
Calculates continuous foliar disease percentage and multi-criteria reliability scoring.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np

from src.morphology import compute_perimeter_and_area
from src.schemas import QualityMetrics, SegmentationResult, SeverityResult, SeverityStatus


class SeverityCalculator:
    """
    Quantifies disease severity percentage and computes multi-factor scientific confidence scores.
    Combines optical clarity (C_q), geometric perimeter-to-area stability (C_s),
    chromatic anomaly contrast (C_t), and optional active robustness ratings (C_r).
    """

    def __init__(
        self,
        w_quality: float = 0.30,
        w_segmentation: float = 0.30,
        w_threshold: float = 0.20,
        w_robustness: float = 0.20,
        boundary_beta: float = 0.05,
        tau_blur: float = 100.0,
        epsilon: float = 1e-6
    ):
        self.w_quality = float(w_quality)
        self.w_segmentation = float(w_segmentation)
        self.w_threshold = float(w_threshold)
        self.w_robustness = float(w_robustness)
        self.boundary_beta = float(boundary_beta)
        self.tau_blur = float(tau_blur)
        self.epsilon = float(epsilon)

        # Normalize weights so sum is 1.0
        total_w = self.w_quality + self.w_segmentation + self.w_threshold + self.w_robustness
        if total_w > 0:
            self.w_quality /= total_w
            self.w_segmentation /= total_w
            self.w_threshold /= total_w
            self.w_robustness /= total_w

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> SeverityCalculator:
        s_cfg = cfg.get("severity", {})
        w_cfg = s_cfg.get("confidence_weights", {})
        q_cfg = cfg.get("quality", {})
        r_cfg = cfg.get("robustness", {})
        return cls(
            w_quality=w_cfg.get("w_quality", 0.30),
            w_segmentation=w_cfg.get("w_segmentation", 0.30),
            w_threshold=w_cfg.get("w_threshold", 0.20),
            w_robustness=w_cfg.get("w_robustness", 0.20),
            boundary_beta=s_cfg.get("boundary_beta", 0.05),
            tau_blur=q_cfg.get("min_laplacian_var", 100.0),
            epsilon=r_cfg.get("epsilon", 1e-6)
        )

    def calculate_severity(
        self,
        segmentation: SegmentationResult,
        quality: QualityMetrics,
        lab_image: np.ndarray,
        lrs_score: Optional[float] = None
    ) -> SeverityResult:
        """
        Calculates severity percentage, unaffected percentage, and composite confidence score.
        """
        if segmentation.leaf_area_px == 0:
            return SeverityResult(
                severity_pct=0.0,
                unaffected_pct=100.0,
                confidence_score=0.0,
                status=SeverityStatus.INVALID_ZERO_LEAF
            )

        # Mathematical severity percentage: S = 100.0 * (A_lesion / A_leaf)
        raw_sev = (float(segmentation.lesion_area_px) / float(segmentation.leaf_area_px)) * 100.0
        severity_pct = float(np.clip(raw_sev, 0.0, 100.0))
        # Ensure exact floating sum to 100.0
        unaffected_pct = float(round(100.0 - severity_pct, 6))

        # 1. Optical Quality Confidence Component C_q
        illum_penalty = quality.glare_pixel_ratio + quality.shadow_pixel_ratio
        c_illum = float(np.clip(1.0 - illum_penalty, 0.0, 1.0))
        c_blur = float(min(1.0, quality.laplacian_variance / max(self.tau_blur, self.epsilon)))
        c_q = c_illum * c_blur

        # 2. Geometric Shape Stability Component C_s
        perimeter, leaf_area = compute_perimeter_and_area(segmentation.leaf_mask)
        if leaf_area > 0:
            isoperimetric_quotient = perimeter / (math.sqrt(float(leaf_area)) + self.epsilon)
            c_s = float(math.exp(-self.boundary_beta * isoperimetric_quotient))
        else:
            c_s = 0.0
        c_s = float(np.clip(c_s, 0.0, 1.0))

        # 3. Chromatic Anomaly Contrast Component C_t
        if segmentation.lesion_area_px > 0:
            a_chan = lab_image[:, :, 1].astype(np.float64)
            lesion_a_vals = a_chan[segmentation.lesion_mask == 255]
            mu_lesion_a = float(np.mean(lesion_a_vals))
            chroma_contrast = mu_lesion_a - segmentation.healthy_mu_a
            mad_spread = 3.0 * (segmentation.healthy_mad_a + segmentation.healthy_mad_b) + self.epsilon
            c_t = float(np.clip(chroma_contrast / mad_spread, 0.0, 1.0))
        else:
            # When no lesions are present, confidence in the healthy label is high
            c_t = 1.0

        # 4. Robustness Factor C_r
        if lrs_score is not None:
            c_r = float(np.clip(lrs_score / 100.0, 0.0, 1.0))
        else:
            c_r = 1.0

        # Composite confidence score
        raw_confidence = 100.0 * (
            self.w_quality * c_q +
            self.w_segmentation * c_s +
            self.w_threshold * c_t +
            self.w_robustness * c_r
        )
        confidence_score = float(np.clip(raw_confidence, 0.0, 100.0))

        # Severity status determination
        if confidence_score < 25.0:
            status = SeverityStatus.LOW_CONFIDENCE
        elif segmentation.lesion_area_px == 0:
            status = SeverityStatus.NO_LESION_DETECTED
        else:
            status = SeverityStatus.LESION_DETECTED

        return SeverityResult(
            severity_pct=severity_pct,
            unaffected_pct=unaffected_pct,
            confidence_score=confidence_score,
            status=status
        )
