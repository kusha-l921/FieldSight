"""
Unit tests for strict typed schemas and runtime invariants.
"""

import numpy as np
import pytest
from src.schemas import (
    ProcessingError,
    ProcessingStatus,
    QualityMetrics,
    QualityStatus,
    SegmentationResult,
    SeverityResult,
    SeverityStatus,
    PerformanceMetrics,
    PerturbationResult,
    RobustnessReport,
    make_array_readonly,
)


def test_quality_metrics_validation():
    # Valid quality metric
    qm = QualityMetrics(
        status=QualityStatus.EXCELLENT,
        mean_lightness=55.0,
        std_lightness=12.0,
        glare_pixel_ratio=0.01,
        shadow_pixel_ratio=0.04,
        laplacian_variance=250.0,
        is_valid_for_processing=True,
        diagnostic_message="Valid test"
    )
    assert qm.status == QualityStatus.EXCELLENT

    # Invalid range tests
    with pytest.raises(ValueError):
        QualityMetrics(
            status=QualityStatus.EXCELLENT,
            mean_lightness=105.0,  # > 100
            std_lightness=12.0,
            glare_pixel_ratio=0.01,
            shadow_pixel_ratio=0.04,
            laplacian_variance=250.0,
            is_valid_for_processing=True,
            diagnostic_message="Invalid lightness"
        )

    with pytest.raises(ValueError):
        QualityMetrics(
            status=QualityStatus.EXCELLENT,
            mean_lightness=50.0,
            std_lightness=-1.0,  # Negative std
            glare_pixel_ratio=0.01,
            shadow_pixel_ratio=0.04,
            laplacian_variance=250.0,
            is_valid_for_processing=True,
            diagnostic_message="Negative std"
        )


def test_segmentation_result_invariants():
    h, w = 100, 100
    leaf = np.zeros((h, w), dtype=np.uint8)
    leaf[20:80, 20:80] = 255  # 60x60 = 3600 px
    
    lesion = np.zeros((h, w), dtype=np.uint8)
    lesion[30:40, 30:40] = 255  # 10x10 = 100 px

    seg = SegmentationResult(
        leaf_mask=leaf,
        lesion_mask=lesion,
        leaf_area_px=3600,
        lesion_area_px=100,
        healthy_mu_a=115.0,
        healthy_mad_a=3.5,
        healthy_mu_b=145.0,
        healthy_mad_b=4.2
    )
    assert seg.leaf_area_px == 3600
    assert seg.lesion_area_px == 100

    # Test read-only array protection
    with pytest.raises(ValueError):
        seg.leaf_mask[0, 0] = 255

    # Test broken invariant: Lesion pixel outside leaf mask
    invalid_lesion = lesion.copy()
    invalid_lesion[0, 0] = 255  # outside leaf (20:80, 20:80)
    with pytest.raises(ValueError, match="Invariant broken"):
        SegmentationResult(
            leaf_mask=leaf,
            lesion_mask=invalid_lesion,
            leaf_area_px=3600,
            lesion_area_px=101,
            healthy_mu_a=115.0,
            healthy_mad_a=3.5,
            healthy_mu_b=145.0,
            healthy_mad_b=4.2
        )


def test_severity_result_validation():
    # Valid severity
    sev = SeverityResult(
        severity_pct=15.5,
        unaffected_pct=84.5,
        confidence_score=92.0,
        status=SeverityStatus.LESION_DETECTED
    )
    assert sev.severity_pct + sev.unaffected_pct == pytest.approx(100.0)

    # Inconsistent sum
    with pytest.raises(ValueError, match="must sum to 100.0"):
        SeverityResult(
            severity_pct=20.0,
            unaffected_pct=70.0,
            confidence_score=90.0,
            status=SeverityStatus.LESION_DETECTED
        )


def test_performance_metrics_validation():
    perf = PerformanceMetrics(
        latency_mean_ms=32.0,
        latency_median_ms=30.0,
        latency_p95_ms=38.0,
        fps=31.25,
        peak_ram_mb=145.0,
        cpu_utilization_pct=18.5
    )
    assert perf.fps > 0

    with pytest.raises(ValueError):
        PerformanceMetrics(
            latency_mean_ms=-5.0,
            latency_median_ms=30.0,
            latency_p95_ms=38.0,
            fps=31.25,
            peak_ram_mb=145.0,
            cpu_utilization_pct=18.5
        )
