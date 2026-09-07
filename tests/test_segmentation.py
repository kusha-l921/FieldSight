"""
Unit tests for Leaf and Lesion Segmentation algorithms.
"""

import cv2
import numpy as np
import pytest
from experiments.dataset_loader import generate_synthetic_leaf
from src.leaf_segmentation import LeafSegmenter
from src.lesion_segmentation import LesionSegmenter
from src.morphology import fill_holes, filter_small_components, compute_perimeter_and_area


@pytest.fixture
def leaf_segmenter():
    return LeafSegmenter(alpha_b_weight=0.5, morph_close_kernel=5, min_leaf_area_ratio=0.05)


@pytest.fixture
def lesion_segmenter():
    return LesionSegmenter(mad_scale_factor=1.4826, chroma_threshold_multiplier=2.5, morph_open_kernel=3)


def test_leaf_segmentation(leaf_segmenter):
    # Synthesize leaf image
    img, gt_leaf, _, _ = generate_synthetic_leaf(severity_target_pct=10.0, seed=42)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    leaf_mask, leaf_area, is_valid = leaf_segmenter.segment_leaf(img, lab_image=lab)
    assert is_valid is True
    assert leaf_area > 1000

    # Overlap with ground truth should be high (> 0.85 IoU)
    intersection = np.count_nonzero((leaf_mask == 255) & (gt_leaf == 255))
    union = np.count_nonzero((leaf_mask == 255) | (gt_leaf == 255))
    iou = intersection / union
    assert iou > 0.85


def test_healthy_leaf_lesion_segmentation(leaf_segmenter, lesion_segmenter):
    # Completely healthy leaf (0% lesions)
    img, _, _, _ = generate_synthetic_leaf(severity_target_pct=0.0, seed=123)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    leaf_mask, leaf_area, is_valid = leaf_segmenter.segment_leaf(img, lab_image=lab)
    seg_res = lesion_segmenter.segment_lesions(lab, leaf_mask)

    assert seg_res.leaf_area_px == leaf_area
    # On completely healthy leaf, segmented lesion area should be 0 or tiny noise (< 0.5%)
    lesion_ratio = seg_res.lesion_area_px / max(seg_res.leaf_area_px, 1)
    assert lesion_ratio < 0.01


def test_diseased_leaf_lesion_segmentation(leaf_segmenter, lesion_segmenter):
    # Diseased leaf (~20% severity)
    img, gt_leaf, gt_lesion, exact_sev = generate_synthetic_leaf(severity_target_pct=20.0, seed=456)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    leaf_mask, _, _ = leaf_segmenter.segment_leaf(img, lab_image=lab)
    seg_res = lesion_segmenter.segment_lesions(lab, leaf_mask)

    assert seg_res.lesion_area_px > 0
    assert seg_res.lesion_area_px <= seg_res.leaf_area_px
    assert np.all((seg_res.lesion_mask == 255) <= (seg_res.leaf_mask == 255))


def test_morphology_utilities():
    # Hole filling test
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 255
    mask[40:50, 40:50] = 0  # 10x10 hole
    
    filled = fill_holes(mask)
    assert np.count_nonzero(filled == 255) == 60 * 60

    # Small component filtering
    noisy = mask.copy()
    noisy[5, 5] = 255  # 1 px noise
    filtered = filter_small_components(noisy, min_area_px=10)
    assert filtered[5, 5] == 0
    assert np.count_nonzero(filtered == 255) == np.count_nonzero(mask == 255)
