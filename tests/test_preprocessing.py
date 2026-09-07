"""
Unit tests for CIELAB color space decoupling and CLAHE preprocessing.
"""

import numpy as np
import pytest
from src.preprocessing import IlluminationNormalizer


@pytest.fixture
def normalizer():
    return IlluminationNormalizer(clahe_clip_limit=2.0, clahe_tile_grid_size=(8, 8))


def test_preprocessing_pipeline(normalizer):
    # Create test BGR image with artificial shadow gradient
    h, w = 128, 128
    img = np.ones((h, w, 3), dtype=np.uint8) * 100
    # Add green foliage hue: B=40, G=140, R=50
    img[:, :, 0] = 40
    img[:, :, 1] = 140
    img[:, :, 2] = 50

    lab_norm, l_norm, bgr_norm = normalizer.normalize(img)

    assert lab_norm.shape == (h, w, 3)
    assert l_norm.shape == (h, w)
    assert bgr_norm.shape == (h, w, 3)
    assert lab_norm.dtype == np.uint8
    assert l_norm.dtype == np.uint8
    assert bgr_norm.dtype == np.uint8


def test_preprocessing_invalid_input(normalizer):
    with pytest.raises(ValueError):
        normalizer.normalize(None)
    with pytest.raises(ValueError):
        normalizer.normalize(np.zeros((0, 0, 3), dtype=np.uint8))
