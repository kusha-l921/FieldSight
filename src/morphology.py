"""
Morphological operations and connected components utilities for FieldSight-Lite.
Provides hole filling, small component filtering, perimeter computation, and spatial cleanup.
"""

from __future__ import annotations

from typing import Tuple
import cv2
import numpy as np


def fill_holes(binary_mask: np.ndarray) -> np.ndarray:
    """
    Fills internal holes within foreground objects in a binary mask {0, 255}.
    Uses contour hierarchy and safe boundary drawing to preserve exterior geometry
    without leaking to canvas boundaries when mask touches the border.
    """
    if binary_mask is None or binary_mask.size == 0:
        return binary_mask

    mask = (binary_mask > 0).astype(np.uint8) * 255
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or hierarchy is None:
        return mask

    filled_mask = np.zeros_like(mask)
    # Draw all top-level (external) contours filled
    for i in range(len(contours)):
        if hierarchy[0][i][3] == -1:  # No parent -> external contour
            cv2.drawContours(filled_mask, contours, i, 255, -1)

    return filled_mask



def filter_small_components(binary_mask: np.ndarray, min_area_px: int) -> np.ndarray:
    """
    Removes connected components smaller than min_area_px.
    """
    if binary_mask is None or binary_mask.size == 0 or min_area_px <= 1:
        return binary_mask

    mask = (binary_mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    output_mask = np.zeros_like(binary_mask, dtype=np.uint8)
    for label_idx in range(1, num_labels):
        area = stats[label_idx, cv2.CC_STAT_AREA]
        if area >= min_area_px:
            output_mask[labels == label_idx] = 255

    return output_mask


def get_largest_component(binary_mask: np.ndarray) -> np.ndarray:
    """
    Isolates the single largest connected component in a binary mask.
    """
    if binary_mask is None or binary_mask.size == 0:
        return binary_mask

    mask = (binary_mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    if num_labels <= 1:
        return np.zeros_like(binary_mask, dtype=np.uint8)

    # Exclude background (index 0)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = np.argmax(areas) + 1

    output_mask = np.zeros_like(binary_mask, dtype=np.uint8)
    output_mask[labels == largest_label] = 255
    return output_mask


def compute_perimeter_and_area(binary_mask: np.ndarray) -> Tuple[float, int]:
    """
    Calculates total boundary perimeter and active pixel area for foreground components.
    """
    if binary_mask is None or binary_mask.size == 0:
        return 0.0, 0

    area_px = int(np.count_nonzero(binary_mask == 255))
    if area_px == 0:
        return 0.0, 0

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = sum(cv2.arcLength(cnt, closed=True) for cnt in contours)
    return float(perimeter), area_px


def apply_morphological_close(binary_mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Applies morphological closing (dilation then erosion) with an elliptical kernel."""
    if kernel_size <= 1 or binary_mask is None or binary_mask.size == 0:
        return binary_mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)


def apply_morphological_open(binary_mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Applies morphological opening (erosion then dilation) with an elliptical kernel."""
    if kernel_size <= 1 or binary_mask is None or binary_mask.size == 0:
        return binary_mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
