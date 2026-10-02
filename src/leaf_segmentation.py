"""
Illumination-invariant leaf blade segmentation for FieldSight-Lite.
Extracts healthy and diseased foliar tissue from background using adaptive colorimetric thresholds.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np

from src.morphology import apply_morphological_close, fill_holes, get_largest_component


class LeafSegmenter:
    """
    Segments leaf lamina from complex agricultural or laboratory backgrounds.
    Combines CIELAB chromaticity cues (low a*, elevated b*) with adaptive Otsu thresholding.
    """

    def __init__(
        self,
        alpha_b_weight: float = 0.5,
        morph_close_kernel: int = 5,
        min_leaf_area_ratio: float = 0.05
    ):
        self.alpha_b_weight = float(alpha_b_weight)
        self.morph_close_kernel = int(morph_close_kernel)
        self.min_leaf_area_ratio = float(min_leaf_area_ratio)

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> LeafSegmenter:
        l_cfg = cfg.get("leaf_segmentation", {})
        return cls(
            alpha_b_weight=l_cfg.get("alpha_b_weight", 0.5),
            morph_close_kernel=l_cfg.get("morph_close_kernel", 5),
            min_leaf_area_ratio=l_cfg.get("min_leaf_area_ratio", 0.05)
        )

    def segment_leaf(
        self,
        bgr_image: np.ndarray,
        lab_image: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, int, bool]:
        """
        Extracts binary leaf mask M_leaf from image.
        Returns:
            leaf_mask: uint8 array [H, W] with values in {0, 255}
            leaf_area_px: count of active foreground pixels
            is_valid_leaf: True if leaf area meets min_leaf_area_ratio, else False
        """
        if bgr_image is None or bgr_image.size == 0:
            return np.zeros((1, 1), dtype=np.uint8), 0, False

        h, w = bgr_image.shape[:2]
        total_px = h * w

        if lab_image is None:
            lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
        else:
            lab = lab_image

        l_chan, a_chan, b_chan = cv2.split(lab)
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        h_chan, s_chan, v_chan = cv2.split(hsv)

        b = bgr_image[:, :, 0].astype(np.float32)
        g = bgr_image[:, :, 1].astype(np.float32)
        r = bgr_image[:, :, 2].astype(np.float32)
        exg = 2.0 * g - r - b

        # Multi-cue foliar biological signature:
        # 1. Chlorophyll green: LAB a* <= 128 or ExG > 0.0
        # 2. Necrotic foliar tissue: a* <= 150, b* > 115, L* > 25 (retains necrotic spots inside blade)
        # 3. Chlorotic / yellowish tissue: HSV saturation > 30, b* > 128, a* <= 135
        is_green = (a_chan <= 128) | (exg > 0.0)
        is_necrotic = (a_chan <= 150) & (b_chan > 115) & (l_chan > 25)
        is_chlorotic = (s_chan > 30) & (b_chan > 128) & (a_chan <= 135)
        canopy_candidate = is_green | is_necrotic | is_chlorotic
        foliage_count = int(np.count_nonzero(canopy_candidate))

        # If image lacks sufficient foliar candidate pixels, reject as NO_LEAF
        if foliage_count < (self.min_leaf_area_ratio * total_px):
            zero_mask = np.zeros((h, w), dtype=np.uint8)
            return zero_mask, 0, False

        # Vegetation chromatic saliency map:
        greenness = (255.0 - a_chan.astype(np.float32))
        yellowness = b_chan.astype(np.float32)

        saliency = (1.0 - self.alpha_b_weight) * greenness + self.alpha_b_weight * yellowness
        saliency_norm = cv2.normalize(saliency, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Otsu's adaptive global thresholding on foliage saliency
        _, raw_mask = cv2.threshold(saliency_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Filter with foliar chromatic constraints
        combined_mask = cv2.bitwise_and(raw_mask, canopy_candidate.astype(np.uint8) * 255)

        # If combined mask is too sparse or empty, fallback to foliar signature or ExG + Saturation thresholding
        if np.count_nonzero(combined_mask) < (self.min_leaf_area_ratio * total_px):
            combined_mask = (canopy_candidate.astype(np.uint8) * 255)
            if np.count_nonzero(combined_mask) < (self.min_leaf_area_ratio * total_px):
                exg_norm = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                _, exg_thresh = cv2.threshold(exg_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                combined_mask = cv2.bitwise_and(exg_thresh, (s_chan > 20).astype(np.uint8) * 255)

        # 1. Morphological closing with kernel_size=7 to bridge venation gaps and fine fissures
        closed_mask = apply_morphological_close(combined_mask, kernel_size=7)

        # 2. Isolate largest connected component (the primary leaf)
        primary_leaf = get_largest_component(closed_mask)

        # 3. Internal Hole Closing Only:
        # Fill strictly enclosed interior contours without modifying or inflating the outer leaf boundary
        leaf_mask = fill_holes(primary_leaf)
        leaf_area_px = int(np.count_nonzero(leaf_mask == 255))

        # Fallback to GrabCut if canopy mask is nearly full canvas (> 0.98)
        if leaf_area_px > 0.98 * total_px:
            rect = (int(w * 0.05), int(h * 0.05), int(w * 0.90), int(h * 0.90))
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            grab_mask = np.zeros((h, w), np.uint8)
            try:
                cv2.grabCut(bgr_image, grab_mask, rect, bgd_model, fgd_model, 2, cv2.GC_INIT_WITH_RECT)
                grab_leaf = np.where((grab_mask == cv2.GC_FGD) | (grab_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
                grab_leaf = get_largest_component(fill_holes(grab_leaf))
                grab_area = int(np.count_nonzero(grab_leaf == 255))
                if (self.min_leaf_area_ratio * total_px) <= grab_area <= (0.95 * total_px):
                    leaf_mask = grab_leaf
                    leaf_area_px = grab_area
            except Exception:
                pass

        # Biological validation: verify candidate component contains foliar tissue (chlorophyll or necrosis)
        if leaf_area_px > 0:
            foliar_in_mask = canopy_candidate[leaf_mask == 255]
            chlorophyll_in_mask = is_green[leaf_mask == 255]
            foliar_fraction = float(np.mean(foliar_in_mask))
            chlorophyll_fraction = float(np.mean(chlorophyll_in_mask))
            mean_exg = float(np.mean(exg[leaf_mask == 255]))
            mean_a = float(np.mean(a_chan[leaf_mask == 255]))
            
            # Reject if mask lacks chlorophyll/foliage signature or has non-vegetative background statistics
            if (chlorophyll_fraction < 0.05 and foliar_fraction < 0.20) or (mean_exg < 3.0 and mean_a > 128.0):
                # Component is non-vegetative background
                leaf_mask = np.zeros((h, w), dtype=np.uint8)
                leaf_area_px = 0

        is_valid_leaf = (leaf_area_px >= (self.min_leaf_area_ratio * total_px))

        if not is_valid_leaf:
            leaf_mask = np.zeros((h, w), dtype=np.uint8)
            leaf_area_px = 0

        return leaf_mask, leaf_area_px, is_valid_leaf

