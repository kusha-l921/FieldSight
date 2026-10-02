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

        # 2. Pass 1: Biological Chlorophyll Reference Anchoring
        global_med_a = float(np.median(a_leaf))
        std_a = float(np.std(a_leaf))
        q25_a = float(np.percentile(a_leaf, 25))
        q10_a = float(np.percentile(a_leaf, 10))
        q15_a = float(np.percentile(a_leaf, 15))

        # Dynamic Anchoring Guard:
        # Only activate chlorophyll quantile anchoring if the leaf exhibits high variance (std_a > 6.0)
        # or widespread chlorophyll degradation (q25_a > 115.0 or global_med_a > 118.0).
        # If the leaf is uniformly green (std_a < 4.0 and global_med_a <= 118.0), use the standard global median to prevent over-segmentation.
        is_high_variance = (std_a > 6.0)
        is_chlorosis_shift = (q25_a > 115.0 or global_med_a > 118.0)

        if is_high_variance or is_chlorosis_shift:
            if q10_a > 122.0:
                # Canopy is nearly completely chlorotic/necrotic with no surviving green pixels; clamp to standard healthy chlorophyll prior
                mu_a_tilde = 112.0
                mu_b_tilde = 145.0
                mad_a_init = 6.0
                mad_b_init = 8.0
            else:
                # Anchor strictly to lowest 10th-to-15th percentile of a* (the surviving green veins)
                anchor_idx = a_leaf <= q15_a
                if np.count_nonzero(anchor_idx) > 10:
                    a_anchor = a_leaf[anchor_idx]
                    b_anchor = b_leaf[anchor_idx]
                    mu_a_tilde = float(np.median(a_anchor))
                    mad_a_init = float(self.mad_scale_factor * np.median(np.abs(a_anchor - mu_a_tilde)))
                    mu_b_tilde = float(np.median(b_anchor))
                    mad_b_init = float(self.mad_scale_factor * np.median(np.abs(b_anchor - mu_b_tilde)))
                else:
                    mu_a_tilde = 112.0
                    mu_b_tilde = 145.0
                    mad_a_init = 6.0
                    mad_b_init = 8.0
        else:
            mu_a_tilde = global_med_a
            mad_a_init = float(self.mad_scale_factor * np.median(np.abs(a_leaf - mu_a_tilde)))
            mu_b_tilde = float(np.median(b_leaf))
            mad_b_init = float(self.mad_scale_factor * np.median(np.abs(b_leaf - mu_b_tilde)))

        # In discrete 8-bit CIELAB space, enforce realistic biological floor for intra-leaf variation
        mad_a_init = max(mad_a_init, 2.5)
        mad_b_init = max(mad_b_init, 2.5)

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

        # Floor MAD to account for discrete color quantization and natural intra-foliar shade variation
        mad_a = max(mad_a, 2.5)
        mad_b = max(mad_b, 2.5)

        # 5. Chromatic Distance Score D_chroma(x, y)
        diff_a = (a_chan - mu_a) / (mad_a + self.epsilon)
        diff_b = (b_chan - mu_b) / (mad_b + self.epsilon)
        d_chroma = np.sqrt(diff_a ** 2 + diff_b ** 2)

        # 6. Candidate Lesion Mask:
        # Detect lesions where pixels deviate positively from the anchored green baseline:
        # - Necrosis/Chlorosis: elevated a* relative to green anchor (a* - mu_a > 2.0 * MAD_a)
        # - Yellowing/Bronzing: elevated b* drift (b* - mu_b > 2.2 * MAD_b) and (a* > mu_a + 0.5 * MAD_a)
        is_anomalous = (d_chroma >= self.chroma_threshold_multiplier)
        is_necrotic = (a_chan > (mu_a + 2.0 * mad_a))
        is_chlorotic = ((b_chan - mu_b) > (2.2 * mad_b)) & (a_chan > (mu_a + 0.5 * mad_a))
        is_pathological_shift = is_necrotic | is_chlorotic

        # 7. Canopy Border Erosion Buffer:
        # Natural leaf serrations and background edge gradients create false chromatic anomalies.
        # Erode leaf_mask by a 3x3 kernel. Pixels within the 2-3px outer boundary are suppressed
        # unless they belong to an anomaly cluster larger than 50 pixels.
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        leaf_mask_eroded = cv2.erode(leaf_mask, erode_kernel, iterations=1)

        raw_lesion_mask = (is_anomalous & is_pathological_shift & (leaf_mask == 255)).astype(np.uint8) * 255

        # 8. Morphological cleanup and minimum anomaly cluster filtering
        opened_mask = apply_morphological_open(raw_lesion_mask, kernel_size=self.morph_open_kernel)
        
        # Filter small noise specs (< 30 pixels or configured min_lesion_component_px)
        effective_min_px = max(self.min_lesion_component_px, 30)
        filtered_mask = filter_small_components(opened_mask, min_area_px=effective_min_px)

        # Suppress boundary artifacts: only retain border pixels if part of larger anomaly cluster (> 50px)
        # Components fully inside leaf_mask_eroded are safe; for components touching border, keep only if area >= 50px
        if np.count_nonzero(filtered_mask) > 0:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((filtered_mask > 0).astype(np.uint8), connectivity=8)
            cleaned_lesion = np.zeros_like(filtered_mask)
            for lbl in range(1, num_labels):
                area = stats[lbl, cv2.CC_STAT_AREA]
                comp_mask = (labels == lbl)
                # If component is smaller than 50px and intersects the outer boundary (not strictly in eroded mask), drop it
                touches_outer_edge = np.any(comp_mask & (leaf_mask_eroded == 0))
                if touches_outer_edge and area < 50:
                    continue
                cleaned_lesion[comp_mask] = 255
            filtered_mask = cleaned_lesion

        # 9. Enforce strict mathematical invariant: M_lesion subset of M_leaf
        lesion_mask = cv2.bitwise_and(filtered_mask, leaf_mask)
        lesion_area_px = int(np.count_nonzero(lesion_mask == 255))

        # 10. Low-variance healthy leaf guard:
        # On healthy leaves with low chromatic variance (std_a < 4.0), suppress subtle false-positive noise (< 50 px)
        if std_a < 4.0 and lesion_area_px < 50:
            lesion_mask = np.zeros((h, w), dtype=np.uint8)
            lesion_area_px = 0

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
