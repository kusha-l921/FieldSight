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


def test_dark_background_canopy_segmentation(leaf_segmenter):
    # Create dark background image with central foliage
    h, w = 300, 300
    img = np.random.randint(10, 30, (h, w, 3), dtype=np.uint8)
    cv2.ellipse(img, (150, 150), (90, 60), 20, 0, 360, (30, 145, 50), -1)
    
    leaf_mask, leaf_area, is_valid = leaf_segmenter.segment_leaf(img)
    assert is_valid is True
    assert leaf_area > 0
    # Must NOT be full canvas
    assert leaf_area < 0.90 * (h * w)


def test_high_severity_lesion_anchoring(leaf_segmenter, lesion_segmenter):
    # Simulate high severity leaf with >50% necrosis
    h, w = 400, 400
    img = np.ones((h, w, 3), dtype=np.uint8) * 210
    cv2.ellipse(img, (200, 200), (120, 80), 0, 0, 360, (35, 140, 60), -1) # Healthy green base
    cv2.ellipse(img, (220, 200), (85, 60), 0, 0, 360, (30, 70, 160), -1) # 50%+ necrotic region

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    leaf_m, l_area, is_valid = leaf_segmenter.segment_leaf(img, lab_image=lab)
    assert is_valid is True
    
    seg_res = lesion_segmenter.segment_lesions(lab, leaf_m)
    sev_pct = (seg_res.lesion_area_px / seg_res.leaf_area_px) * 100.0
    # Severity must be detected in expected high range (40% - 65%) and NOT invert to <1%
    assert 40.0 <= sev_pct <= 65.0


def test_chlorosis_interveinal_bronzing_anchoring(leaf_segmenter, lesion_segmenter):
    # Simulate a leaf with ~50% interveinal chlorosis/bronzing
    h, w = 400, 400
    img = np.ones((h, w, 3), dtype=np.uint8) * 210
    # Base green lamina
    cv2.ellipse(img, (200, 200), (120, 80), 0, 0, 360, (35, 140, 55), -1)
    # Interveinal chlorosis / bronzing patches covering ~50%
    cv2.ellipse(img, (160, 180), (45, 30), 20, 0, 360, (45, 130, 120), -1)
    cv2.ellipse(img, (240, 180), (45, 30), -20, 0, 360, (45, 130, 120), -1)
    cv2.ellipse(img, (200, 230), (50, 25), 0, 0, 360, (45, 130, 120), -1)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    leaf_m, l_area, is_valid = leaf_segmenter.segment_leaf(img, lab_image=lab)
    assert is_valid is True

    seg_res = lesion_segmenter.segment_lesions(lab, leaf_m)
    sev_pct = (seg_res.lesion_area_px / seg_res.leaf_area_px) * 100.0
    # Must capture interveinal chlorosis and yield severity in 40% - 65% range
    assert 40.0 <= sev_pct <= 65.0


def test_necrotic_canopy_core_retention(leaf_segmenter, lesion_segmenter):
    # Simulate Image 1: Leaf blade with a large dead brown/necrotic central lesion
    h, w = 400, 400
    img = np.ones((h, w, 3), dtype=np.uint8) * 220 # neutral light background
    # Outer green leaf lamina
    cv2.ellipse(img, (200, 200), (130, 90), 0, 0, 360, (40, 145, 60), -1)
    # Central necrotic brown lesion (a* ~ 140, b* ~ 135, L* ~ 80)
    cv2.ellipse(img, (200, 200), (60, 45), 15, 0, 360, (35, 65, 135), -1)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    leaf_m, l_area, is_valid = leaf_segmenter.segment_leaf(img, lab_image=lab)
    assert is_valid is True

    # Ensure central necrotic core is part of leaf_mask (no doughnut hole)
    center_region = leaf_m[180:220, 180:220]
    assert np.all(center_region == 255), "Necrotic center should be fully retained in leaf mask"

    # Segment lesions
    seg_res = lesion_segmenter.segment_lesions(lab, leaf_m)
    sev_pct = (seg_res.lesion_area_px / seg_res.leaf_area_px) * 100.0
    # Severity should be detected in ~20-35% range and not suppressed to <10%
    assert 20.0 <= sev_pct <= 35.0


def test_necrotic_perimeter_bay_closure(leaf_segmenter, lesion_segmenter):
    # Simulate a leaf where necrotic necrosis touches or breaches the outer perimeter
    h, w = 400, 400
    img = np.ones((h, w, 3), dtype=np.uint8) * 210
    # Outer green leaf blade
    cv2.ellipse(img, (200, 200), (130, 90), 0, 0, 360, (40, 145, 60), -1)
    # Necrotic lesion touching blade edge
    cv2.ellipse(img, (240, 200), (70, 50), 0, 0, 360, (35, 65, 135), -1)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    leaf_m, l_area, is_valid = leaf_segmenter.segment_leaf(img, lab_image=lab)
    assert is_valid is True

    # Check that canopy mask has no internal cavities or perimeter cutout notches in necrotic zone
    bay_region = leaf_m[180:220, 220:260]
    assert np.all(bay_region == 255), "Perimeter necrotic bay must be solid without void cutout"

    seg_res = lesion_segmenter.segment_lesions(lab, leaf_m)
    sev_pct = (seg_res.lesion_area_px / seg_res.leaf_area_px) * 100.0
    # Realistic biological severity (~25% - 40%)
    assert 25.0 <= sev_pct <= 40.0


def test_clean_healthy_leaf_zero_false_positive(leaf_segmenter, lesion_segmenter):
    # Simulate Image 2: Clean healthy leaf with boundary gradient/serrations
    h, w = 400, 400
    img = np.ones((h, w, 3), dtype=np.uint8) * 215
    # Clean green leaf
    cv2.ellipse(img, (200, 200), (120, 80), 0, 0, 360, (42, 142, 58), -1)
    # Add minor boundary shadow/noise
    cv2.ellipse(img, (200, 200), (120, 80), 0, 0, 360, (30, 120, 45), 2)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    leaf_m, l_area, is_valid = leaf_segmenter.segment_leaf(img, lab_image=lab)
    assert is_valid is True

    seg_res = lesion_segmenter.segment_lesions(lab, leaf_m)
    sev_pct = (seg_res.lesion_area_px / seg_res.leaf_area_px) * 100.0
    # On clean healthy leaf, severity must be < 0.5%
    assert sev_pct < 0.5





