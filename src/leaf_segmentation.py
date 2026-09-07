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

        b = bgr_image[:, :, 0].astype(np.float32)
        g = bgr_image[:, :, 1].astype(np.float32)
        r = bgr_image[:, :, 2].astype(np.float32)
        exg = 2.0 * g - r - b

        # Foliar biological signature:
        # Green plant foliage has a* < 125 (in OpenCV 8-bit LAB, 128 is neutral gray) and positive excess green
        is_foliage_pixel = (a_chan <= 124) | (exg > 10.0)
        foliage_count = int(np.count_nonzero(is_foliage_pixel))

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
        combined_mask = cv2.bitwise_and(raw_mask, is_foliage_pixel.astype(np.uint8) * 255)

        # Morphological closing to seal venation gaps
        closed_mask = apply_morphological_close(combined_mask, kernel_size=self.morph_close_kernel)

        # Fill internal holes
        filled_mask = fill_holes(closed_mask)

        # Isolate largest leaf canopy component
        leaf_mask = get_largest_component(filled_mask)
        leaf_area_px = int(np.count_nonzero(leaf_mask == 255))

        # Biological validation: verify candidate component has foliar chlorophyll characteristics
        if leaf_area_px > 0:
            mean_exg = float(np.mean(exg[leaf_mask == 255]))
            mean_a = float(np.mean(a_chan[leaf_mask == 255]))
            if mean_exg < 8.0 and mean_a > 125.0:
                # Component is non-vegetative background
                leaf_mask = np.zeros((h, w), dtype=np.uint8)
                leaf_area_px = 0

        is_valid_leaf = (leaf_area_px >= (self.min_leaf_area_ratio * total_px))

        if not is_valid_leaf:
            leaf_mask = np.zeros((h, w), dtype=np.uint8)
            leaf_area_px = 0

        return leaf_mask, leaf_area_px, is_valid_leaf
