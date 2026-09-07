"""
Unit tests for pre-flight optical quality assessment.
"""

import cv2
import numpy as np
import pytest
from src.quality import ImageQualityAssessor
from src.schemas import QualityStatus


@pytest.fixture
def quality_assessor():
    return ImageQualityAssessor(
        min_laplacian_var=100.0,
        max_glare_ratio=0.15,
        max_shadow_ratio=0.35,
        glare_l_physical_thresh=96.0,
        shadow_l_physical_thresh=10.0,
        glare_sat_max=30
    )


def test_clear_image_quality(quality_assessor):
    # Textured sharp foliage image
    img = np.random.randint(40, 180, (200, 200, 3), dtype=np.uint8)
    qm = quality_assessor.assess_quality(img)
    assert qm.is_valid_for_processing is True
    assert qm.status in [QualityStatus.EXCELLENT, QualityStatus.ACCEPTABLE]


def test_blurry_image_rejection(quality_assessor):
    # Completely uniform flat image (Laplacian variance = 0)
    flat_img = np.ones((200, 200, 3), dtype=np.uint8) * 128
    qm = quality_assessor.assess_quality(flat_img)
    assert qm.is_valid_for_processing is False
    assert qm.status == QualityStatus.BLURRY


def test_excessive_glare_rejection(quality_assessor):
    # Image with large pure white patch (glare)
    img = np.random.randint(40, 180, (200, 200, 3), dtype=np.uint8)
    # Add 40% pure white specular glare
    img[0:150, 0:150] = 255
    qm = quality_assessor.assess_quality(img)
    assert qm.is_valid_for_processing is False
    assert qm.status == QualityStatus.DEGRADED_GLARE


def test_excessive_shadow_rejection(quality_assessor):
    # Image with 80% deep black shadow
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[0:50, 0:50] = np.random.randint(50, 150, (50, 50, 3), dtype=np.uint8)
    qm = quality_assessor.assess_quality(img)
    assert qm.is_valid_for_processing is False
    assert qm.status in [QualityStatus.DEGRADED_SHADOW, QualityStatus.BLURRY]


def test_invalid_and_corrupt_inputs(quality_assessor):
    assert quality_assessor.assess_quality(None).status == QualityStatus.UNUSABLE
    assert quality_assessor.assess_quality(np.zeros((0, 0, 3), dtype=np.uint8)).status == QualityStatus.UNUSABLE
    assert quality_assessor.assess_quality(np.zeros((10, 10), dtype=np.uint8)).status == QualityStatus.UNUSABLE
