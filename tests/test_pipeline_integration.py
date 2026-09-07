"""
Integration tests for the master FieldSightPipeline coordinator.
"""

import os
import tempfile
import cv2
import numpy as np
import pytest

from experiments.dataset_loader import generate_synthetic_leaf
from src.pipeline import FieldSightPipeline
from src.schemas import ProcessingStatus, SeverityStatus


@pytest.fixture
def pipeline():
    # Use temporary sqlite database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    cfg = {
        "version": "1.0.0",
        "pipeline": {"seed": 42},
        "quality": {
            "min_laplacian_var": 100.0,
            "max_glare_ratio": 0.15,
            "max_shadow_ratio": 0.35,
            "glare_l_physical_thresh": 96.0,
            "shadow_l_physical_thresh": 10.0,
            "glare_sat_max": 30
        },
        "preprocessing": {
            "clahe_clip_limit": 2.0,
            "clahe_tile_grid_size": [8, 8]
        },
        "leaf_segmentation": {
            "alpha_b_weight": 0.5,
            "morph_close_kernel": 5,
            "min_leaf_area_ratio": 0.05
        },
        "lesion_segmentation": {
            "mad_scale_factor": 1.4826,
            "chroma_threshold_multiplier": 2.5,
            "morph_open_kernel": 3,
            "min_lesion_component_px": 5
        },
        "severity": {
            "confidence_weights": {
                "w_quality": 0.30,
                "w_segmentation": 0.30,
                "w_threshold": 0.20,
                "w_robustness": 0.20
            },
            "boundary_beta": 0.05
        },
        "robustness": {
            "epsilon": 1e-6,
            "fri_weights": {
                "w_robustness": 0.40,
                "w_segmentation": 0.30,
                "w_latency": 0.20,
                "w_memory": 0.10
            },
            "latency_baseline_ms": 50.0,
            "memory_baseline_mb": 200.0
        },
        "progression": {
            "stable_max_pct_per_day": 0.5,
            "emerging_max_pct_per_day": 1.5,
            "moderate_max_pct_per_day": 3.5
        },
        "storage": {
            "sqlite_db_path": db_path
        }
    }
    
    pipe = FieldSightPipeline(cfg)
    yield pipe
    if os.path.exists(db_path):
        os.remove(db_path)


def test_end_to_end_diseased_leaf(pipeline):
    img, _, _, exact_sev = generate_synthetic_leaf(severity_target_pct=15.0, seed=777)
    res = pipeline.process_image(
        image_input=img,
        plant_id="TEST_PLANT_1",
        leaf_id="TEST_LEAF_A",
        run_stress_test=True
    )

    assert res.status == ProcessingStatus.SUCCESS
    assert res.quality.is_valid_for_processing is True
    assert res.segmentation is not None
    assert res.severity is not None
    assert res.severity.severity_pct > 0.0
    assert res.severity.unaffected_pct + res.severity.severity_pct == pytest.approx(100.0)
    assert res.overlay_image is not None
    assert res.robustness is not None
    assert res.robustness.lighting_robustness_score > 0.0
    assert res.progression is not None
    assert len(res.progression.observation_history) == 1

    # Test JSON export
    manifest = pipeline.export_result_dict(res)
    assert manifest["status"] == "SUCCESS"
    assert "severity" in manifest
    assert "robustness" in manifest
    assert "performance" in manifest


def test_end_to_end_no_leaf_rejection(pipeline):
    # Pure flat beige background (no leaf)
    blank_bg = np.ones((300, 300, 3), dtype=np.uint8) * 200
    # Add noise to pass blur check
    noise = np.random.randint(-15, 15, (300, 300, 3), dtype=np.int16)
    blank_bg = np.clip(blank_bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    res = pipeline.process_image(blank_bg)
    assert res.status == ProcessingStatus.NO_LEAF
    assert res.segmentation is None
    assert res.severity is None
    assert len(res.errors) > 0
    assert res.errors[0].code == ProcessingStatus.NO_LEAF


def test_end_to_end_blurry_rejection(pipeline):
    # Completely flat image
    flat = np.ones((200, 200, 3), dtype=np.uint8) * 128
    res = pipeline.process_image(flat)
    assert res.status == ProcessingStatus.BLURRY_IMAGE
    assert res.segmentation is None
    assert res.severity is None
