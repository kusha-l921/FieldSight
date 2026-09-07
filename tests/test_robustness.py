"""
Unit tests for synthetic lighting transformations and robustness metrics engine.
"""

import numpy as np
import pytest
from experiments.dataset_loader import generate_synthetic_leaf
from experiments.synthetic_perturb import (
    apply_gamma_shift,
    apply_overexposure,
    apply_shadow_ramp,
    apply_specular_glare,
    apply_underexposure,
    generate_all_perturbations,
)
from src.robustness import RobustnessEngine, compute_mask_iou
from src.schemas import SegmentationResult, SeverityResult, SeverityStatus


@pytest.fixture
def robustness_engine():
    return RobustnessEngine(
        w_robustness=0.40,
        w_segmentation=0.30,
        w_latency=0.20,
        w_memory=0.10,
        latency_baseline_ms=50.0,
        memory_baseline_mb=200.0
    )


def test_synthetic_transformations():
    img = np.random.randint(50, 200, (128, 128, 3), dtype=np.uint8)

    shadow = apply_shadow_ramp(img)
    glare = apply_specular_glare(img)
    over = apply_overexposure(img)
    under = apply_underexposure(img)
    gamma = apply_gamma_shift(img)

    for trans in [shadow, glare, over, under, gamma]:
        assert trans.shape == img.shape
        assert trans.dtype == np.uint8

    all_perts = generate_all_perturbations(img)
    assert len(all_perts) == 5


def test_mask_iou_computation():
    m1 = np.zeros((100, 100), dtype=np.uint8)
    m2 = np.zeros((100, 100), dtype=np.uint8)
    # Both empty -> IoU = 1.0
    assert compute_mask_iou(m1, m2) == 1.0

    m1[10:50, 10:50] = 255
    m2[10:50, 10:50] = 255
    # Exact overlap -> IoU = 1.0
    assert compute_mask_iou(m1, m2) == 1.0

    m2 = np.zeros((100, 100), dtype=np.uint8)
    m2[50:90, 50:90] = 255
    # Disjoint -> IoU = 0.0
    assert compute_mask_iou(m1, m2) == 0.0


def test_robustness_evaluation(robustness_engine):
    img, gt_leaf, gt_les, sev = generate_synthetic_leaf(severity_target_pct=15.0, seed=999)
    leaf_area = int(np.count_nonzero(gt_leaf == 255))
    les_area = int(np.count_nonzero(gt_les == 255))

    base_seg = SegmentationResult(
        leaf_mask=gt_leaf,
        lesion_mask=gt_les,
        leaf_area_px=leaf_area,
        lesion_area_px=les_area,
        healthy_mu_a=118.0,
        healthy_mad_a=3.0,
        healthy_mu_b=142.0,
        healthy_mad_b=3.5
    )
    base_sev = SeverityResult(
        severity_pct=sev,
        unaffected_pct=100.0 - sev,
        confidence_score=95.0,
        status=SeverityStatus.LESION_DETECTED
    )

    def dummy_segment(perturbed_img):
        # Slightly perturbed segmentation
        return base_seg, base_sev

    report = robustness_engine.evaluate_robustness(
        original_bgr=img,
        base_segmentation=base_seg,
        base_severity=base_sev,
        segment_fn=dummy_segment
    )

    assert 0.0 <= report.lighting_robustness_score <= 100.0
    assert 0.0 <= report.field_robustness_index <= 100.0
    assert report.mean_severity_drift >= 0.0
    assert len(report.perturbation_breakdown) == 5
