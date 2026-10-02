"""
Illumination-robust colorimetric preprocessing engine for FieldSight-Lite.
Decouples lightness from chromaticity using CIELAB color space and local CLAHE.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple
import cv2
import numpy as np


class IlluminationNormalizer:
    """
    Normalizes spatial lightness gradients across plant foliage using CIELAB and CLAHE.
    Flattens harsh direct sunlight, cast shadows, and vignetting without distorting chromatic cues.
    """

    def __init__(
        self,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: Tuple[int, int] = (8, 8)
    ):
        self.clahe_clip_limit = float(clahe_clip_limit)
        self.clahe_tile_grid_size = (int(clahe_tile_grid_size[0]), int(clahe_tile_grid_size[1]))
        self._clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_tile_grid_size
        )

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> IlluminationNormalizer:
        p_cfg = cfg.get("preprocessing", {})
        grid_size = p_cfg.get("clahe_tile_grid_size", [8, 8])
        return cls(
            clahe_clip_limit=p_cfg.get("clahe_clip_limit", 2.0),
            clahe_tile_grid_size=(grid_size[0], grid_size[1])
        )

    def normalize(self, bgr_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Processes BGR input image through localized CLAHE on the L*-channel.
        Returns:
            lab_normalized: uint8 array [H, W, 3] in CIELAB space with equalized L*
            l_channel_norm: uint8 array [H, W] normalized L* channel
            bgr_normalized: uint8 array [H, W, 3] reconstructed BGR image
        """
        if bgr_image is None or bgr_image.size == 0:
            raise ValueError("Input image cannot be None or empty")

        # Convert BGR to CIELAB (OpenCV representation: L: 0-255, a: 0-255, b: 0-255)
        lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)

        # Apply Contrast Limited Adaptive Histogram Equalization strictly to L*
        l_norm = self._clahe.apply(l_chan)

        # Re-merge channels with unmodified chromaticity (a*, b*)
        lab_normalized = cv2.merge([l_norm, a_chan, b_chan])

        # Convert back to BGR for visual inspection / overlays
        bgr_normalized = cv2.cvtColor(lab_normalized, cv2.COLOR_LAB2BGR)

        return lab_normalized, l_norm, bgr_normalized
