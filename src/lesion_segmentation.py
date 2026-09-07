"""
Two-pass dynamic inlier robust statistical lesion segmentation for FieldSight-Lite.
Quantifies symptomatic chlorotic and necrotic tissue using Median Absolute Deviation (MAD) in CIELAB chromaticity space.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np

from src.morphology import apply_morphological_open, filter_small_components
from src.schemas import SegmentationResult


class LesionSegmenter:
    """
    Training-free, statistical anomaly segmentation engine for foliar pathologies.
    Self-calibrates healthy baseline tissue statistics per leaf and isolates necrotic/chlorotic outliers.
    """

    def __init__(
        self,
        mad_scale_factor: float = 1.4826,
        chroma_threshold_multiplier: float = 2.5,
        morph_open_kernel: int = 3,
        min_lesion_component_px: int = 5,
        epsilon: float = 1e-6
    ):
        self.mad_scale_factor = float(mad_scale_factor)
        self.chroma_threshold_multiplier = float(chroma_threshold_multiplier)
        self.morph_open_kernel = int(morph_open_kernel)
        self.min_lesion_component_px = int(min_lesion_component_px)
        self.epsilon = float(epsilon)

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> LesionSegmenter:
        l_cfg = cfg.get("lesion_segmentation", {})
        r_cfg = cfg.get("robustness", {})
        return cls(
            mad_scale_factor=l_cfg.get("mad_scale_factor", 1.4826),
            chroma_threshold_multiplier=l_cfg.get("chroma_threshold_multiplier", 2.5),
            morph_open_kernel=l_cfg.get("morph_open_kernel", 3),
            min_lesion_component_px=l_cfg.get("min_lesion_component_px", 5),
            epsilon=r_cfg.get("epsilon", 1e-6)
        )

    def segment_lesions(
        self,
        lab_image: np.ndarray,
        leaf_mask: np.ndarray
    ) -> SegmentationResult:
        """
        Executes two-pass MAD outlier anomaly segmentation on leaf lamina.
        Returns immutable SegmentationResult satisfying all formal invariants.
        """
        if lab_image is None or leaf_mask is None:
            raise ValueError("Inputs lab_image and leaf_mask cannot be None")

        h, w = lab_image.shape[:2]
        leaf_indices = (leaf_mask == 255)
        leaf_area_px = int(np.count_nonzero(leaf_indices))

        # Handle zero leaf case
        if leaf_area_px == 0:
            zero_mask = np.zeros((h, w), dtype=np.uint8)
            return SegmentationResult(
                leaf_mask=zero_mask,
                lesion_mask=zero_mask,
                leaf_area_px=0,
                lesion_area_px=0,
                healthy_mu_a=0.0,
                healthy_mad_a=0.0,
                healthy_mu_b=0.0,
                healthy_mad_b=0.0
            )

        a_chan = lab_image[:, :, 1].astype(np.float64)
        b_chan = lab_image[:, :, 2].astype(np.float64)

        # 1. Extract foliar pixel chromatic distributions
        a_leaf = a_chan[leaf_indices]
        b_leaf = b_chan[leaf_indices]

        # 2. Pass 1: Initial Medians and MADs
        mu_a_tilde = float(np.median(a_leaf))
        mad_a_init = float(self.mad_scale_factor * np.median(np.abs(a_leaf - mu_a_tilde)))

        mu_b_tilde = float(np.median(b_leaf))
        mad_b_init = float(self.mad_scale_factor * np.median(np.abs(b_leaf - mu_b_tilde)))

        # In discrete 8-bit CIELAB space, enforce realistic biological floor for intra-leaf variation
        mad_a_init = max(mad_a_init, 2.0)
        mad_b_init = max(mad_b_init, 2.0)

        # 3. Isolate healthy inlier reference mask H_ref (Pass 1 inliers)
        # Healthy plant tissue is characterized by lower a* (greener) and moderate b*
        h_ref_mask = (a_leaf <= (mu_a_tilde + 1.5 * mad_a_init)) & (b_leaf <= (mu_b_tilde + 1.5 * mad_b_init))
        h_ref_count = int(np.count_nonzero(h_ref_mask))

        # 4. Pass 2: Recalculate baseline statistics on healthy inliers H_ref
        if h_ref_count > 10:
            a_healthy = a_leaf[h_ref_mask]
            b_healthy = b_leaf[h_ref_mask]
            mu_a = float(np.median(a_healthy))
            mad_a = float(self.mad_scale_factor * np.median(np.abs(a_healthy - mu_a)))
            mu_b = float(np.median(b_healthy))
            mad_b = float(self.mad_scale_factor * np.median(np.abs(b_healthy - mu_b)))
        else:
            mu_a = mu_a_tilde
            mad_a = mad_a_init
            mu_b = mu_b_tilde
            mad_b = mad_b_init

        # Floor MAD to account for discrete color quantization
        mad_a = max(mad_a, 2.0)
        mad_b = max(mad_b, 2.0)

        # 5. Chromatic Distance Score D_chroma(x, y)
        diff_a = (a_chan - mu_a) / (mad_a + self.epsilon)
        diff_b = (b_chan - mu_b) / (mad_b + self.epsilon)
        d_chroma = np.sqrt(diff_a ** 2 + diff_b ** 2)

        # 6. Candidate Lesion Mask:
        # Lesion pixels require D_chroma >= k_thresh AND a* > mu_a (loss of green chlorophyll)
        is_anomalous = (d_chroma >= self.chroma_threshold_multiplier)
        is_necrotic = (a_chan > (mu_a + 0.8 * mad_a))
        is_chlorotic = (a_chan > mu_a) & ((b_chan - mu_b) > (2.5 * mad_b))
        is_pathological_shift = is_necrotic | is_chlorotic

        raw_lesion_mask = (is_anomalous & is_pathological_shift & (leaf_mask == 255)).astype(np.uint8) * 255

        # 7. Morphological cleanup
        opened_mask = apply_morphological_open(raw_lesion_mask, kernel_size=self.morph_open_kernel)
        filtered_mask = filter_small_components(opened_mask, min_area_px=self.min_lesion_component_px)

        # 8. Enforce strict mathematical invariant: M_lesion subset of M_leaf
        lesion_mask = cv2.bitwise_and(filtered_mask, leaf_mask)
        lesion_area_px = int(np.count_nonzero(lesion_mask == 255))

        return SegmentationResult(
            leaf_mask=leaf_mask,
            lesion_mask=lesion_mask,
            leaf_area_px=leaf_area_px,
            lesion_area_px=lesion_area_px,
            healthy_mu_a=mu_a,
            healthy_mad_a=mad_a,
            healthy_mu_b=mu_b,
            healthy_mad_b=mad_b
        )
