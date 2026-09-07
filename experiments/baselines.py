"""
Baseline algorithms for experimental benchmarking against FieldSight-Lite.
Implements standard literature baselines: Naive RGB Thresholding, HSV Color Slicing,
Global Otsu Grayscale Thresholding, and K-Means Color Quantization.
"""

from __future__ import annotations

from typing import Tuple
import cv2
import numpy as np


class RGBThresholdBaseline:
    """
    Baseline 1: Naive RGB channel thresholding.
    Extracts leaf via (G > R and G > B) and lesions via (R > G on leaf).
    """

    def predict(self, bgr_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        h, w = bgr_image.shape[:2]
        b = bgr_image[:, :, 0].astype(np.int16)
        g = bgr_image[:, :, 1].astype(np.int16)
        r = bgr_image[:, :, 2].astype(np.int16)

        # Leaf mask: Green is dominant channel
        leaf_mask = ((g > (r + 10)) & (g > (b + 10))).astype(np.uint8) * 255
        leaf_area = int(np.count_nonzero(leaf_mask == 255))

        if leaf_area == 0:
            return leaf_mask, np.zeros((h, w), dtype=np.uint8), 0.0

        # Lesion mask: Necrotic shift where Red dominates Green within leaf
        lesion_mask = (((r > g) | ((r - b) > 40)) & (leaf_mask == 255)).astype(np.uint8) * 255
        lesion_area = int(np.count_nonzero(lesion_mask == 255))

        sev_pct = float((lesion_area / leaf_area) * 100.0)
        return leaf_mask, lesion_mask, sev_pct


class HSVThresholdBaseline:
    """
    Baseline 2: HSV Color space thresholding.
    Uses fixed hue ranges for foliage (H: 30-90) and necrosis (H: 0-25).
    """

    def predict(self, bgr_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        h, w = bgr_image.shape[:2]
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        # Leaf mask: Foliage green-yellow hues
        leaf_mask = ((hue >= 25) & (hue <= 90) & (sat >= 30) & (val >= 30)).astype(np.uint8) * 255
        leaf_area = int(np.count_nonzero(leaf_mask == 255))

        if leaf_area == 0:
            return leaf_mask, np.zeros((h, w), dtype=np.uint8), 0.0

        # Lesion mask: Orange/brown hues (low hue)
        lesion_mask = (((hue <= 22) | (hue >= 170)) & (sat >= 40) & (leaf_mask == 255)).astype(np.uint8) * 255
        lesion_area = int(np.count_nonzero(lesion_mask == 255))

        sev_pct = float((lesion_area / leaf_area) * 100.0)
        return leaf_mask, lesion_mask, sev_pct


class GlobalOtsuBaseline:
    """
    Baseline 3: Classical Global Otsu on grayscale intensity.
    """

    def predict(self, bgr_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        h, w = bgr_image.shape[:2]
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        
        # Otsu on inverted grayscale for leaf extraction
        _, leaf_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        leaf_area = int(np.count_nonzero(leaf_mask == 255))

        if leaf_area == 0:
            return leaf_mask, np.zeros((h, w), dtype=np.uint8), 0.0

        # Lesion thresholding on leaf pixels
        leaf_pixels = gray[leaf_mask == 255]
        if len(leaf_pixels) > 0:
            thresh_val, _ = cv2.threshold(leaf_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            lesion_mask = ((gray >= thresh_val) & (leaf_mask == 255)).astype(np.uint8) * 255
        else:
            lesion_mask = np.zeros((h, w), dtype=np.uint8)

        lesion_area = int(np.count_nonzero(lesion_mask == 255))
        sev_pct = float((lesion_area / leaf_area) * 100.0)
        return leaf_mask, lesion_mask, sev_pct


class KMeansColorBaseline:
    """
    Baseline 4: Unsupervised K-Means Color Quantization (K=3: Background, Healthy, Lesion).
    """

    def predict(self, bgr_image: np.ndarray, k: int = 3) -> Tuple[np.ndarray, np.ndarray, float]:
        h, w = bgr_image.shape[:2]
        data = bgr_image.reshape((-1, 3)).astype(np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv2.KMEANS_RANDOM_CENTERS
        compactness, labels, centers = cv2.kmeans(data, k, None, criteria, 3, flags)

        labels = labels.flatten()
        centers = centers.astype(np.uint8)

        # Identify healthy center (highest green ratio G / (R + B + 1))
        green_ratios = [float(c[1]) / (float(c[0]) + float(c[2]) + 1.0) for c in centers]
        healthy_idx = int(np.argmax(green_ratios))

        # Identify lesion center (reddest remaining cluster)
        red_ratios = [float(c[2]) / (float(c[1]) + float(c[0]) + 1.0) if i != healthy_idx else -1.0 for i, c in enumerate(centers)]
        lesion_idx = int(np.argmax(red_ratios))

        leaf_mask = (((labels == healthy_idx) | (labels == lesion_idx)).reshape((h, w)).astype(np.uint8)) * 255
        lesion_mask = ((labels == lesion_idx).reshape((h, w)).astype(np.uint8)) * 255

        leaf_area = int(np.count_nonzero(leaf_mask == 255))
        lesion_area = int(np.count_nonzero(lesion_mask == 255))

        if leaf_area == 0:
            return leaf_mask, np.zeros((h, w), dtype=np.uint8), 0.0

        sev_pct = float((lesion_area / leaf_area) * 100.0)
        return leaf_mask, lesion_mask, sev_pct
