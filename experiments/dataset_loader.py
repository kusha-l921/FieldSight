"""
Dataset loading, sample synthesis, and strict Plant-ID split manager for FieldSight-Lite.
Prevents data leakage across experimental splits and generates ground-truth benchmark pairs.
"""

from __future__ import annotations

import glob
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


@dataclass(frozen=True)
class BenchmarkSample:
    sample_id: str
    plant_id: str
    leaf_id: str
    image_bgr: np.ndarray
    gt_leaf_mask: Optional[np.ndarray] = None
    gt_lesion_mask: Optional[np.ndarray] = None
    gt_severity_pct: Optional[float] = None
    metadata: Optional[Dict[str, str]] = None


def generate_synthetic_leaf(
    image_size: Tuple[int, int] = (512, 512),
    severity_target_pct: float = 15.0,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Synthesizes a realistic plant leaf image with known ground-truth leaf and lesion masks.
    Uses procedural elliptical contours, venation structures, chlorophyll gradients,
    and necrotic brown/yellow lesions.
    
    Returns:
        (image_bgr, gt_leaf_mask, gt_lesion_mask, exact_severity_pct)
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    h, w = image_size
    # Neutral agricultural background (light gray/beige workbench with subtle texture)
    bg = np.ones((h, w, 3), dtype=np.uint8) * 210
    noise = np.random.randint(-10, 10, (h, w, 3), dtype=np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 1. Create Leaf Blade Geometry
    center = (w // 2, h // 2)
    axes = (int(w * 0.36), int(h * 0.44))
    angle = random.uniform(-15.0, 15.0)

    gt_leaf_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(gt_leaf_mask, center, axes, angle, 0, 360, 255, -1)

    # Deform contour slightly for natural organic leaf shape
    contours, _ = cv2.findContours(gt_leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        cnt = contours[0]
        # Smooth organic perturbation
        for pt in cnt:
            shift_x = int(np.sin(pt[0][1] * 0.05) * 6)
            shift_y = int(np.cos(pt[0][0] * 0.05) * 6)
            pt[0][0] += shift_x
            pt[0][1] += shift_y
        gt_leaf_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(gt_leaf_mask, [cnt], -1, 255, -1)

    leaf_indices = (gt_leaf_mask == 255)
    leaf_area_px = int(np.count_nonzero(leaf_indices))

    # 2. Render Chlorophyll Foliage Texture
    # Base green: B=35, G=140, R=55 in BGR (corresponds to healthy green)
    leaf_layer = np.zeros((h, w, 3), dtype=np.uint8)
    raw_b = np.random.randint(25, 45, (h, w), dtype=np.uint8)
    raw_g = np.random.randint(125, 155, (h, w), dtype=np.uint8)
    raw_r = np.random.randint(45, 70, (h, w), dtype=np.uint8)
    leaf_layer[:, :, 0] = cv2.GaussianBlur(raw_b, (7, 7), 2.5)
    leaf_layer[:, :, 1] = cv2.GaussianBlur(raw_g, (7, 7), 2.5)
    leaf_layer[:, :, 2] = cv2.GaussianBlur(raw_r, (7, 7), 2.5)

    # Add primary leaf vein
    pt1 = (int(center[0] - axes[0] * 0.8), int(center[1] - axes[1] * 0.8))
    pt2 = (int(center[0] + axes[0] * 0.8), int(center[1] + axes[1] * 0.8))
    cv2.line(leaf_layer, pt1, pt2, (40, 175, 80), 3)

    # 3. Render Pathological Lesions (Necrosis/Chlorosis)
    gt_lesion_mask = np.zeros((h, w), dtype=np.uint8)
    target_lesion_px = int((severity_target_pct / 100.0) * leaf_area_px)

    if target_lesion_px > 0 and leaf_area_px > 0:
        # Scatter circular/elliptical lesions on leaf blade
        num_lesion_clusters = max(1, int(severity_target_pct * 0.6))
        placed_px = 0
        attempts = 0

        # Find leaf coordinate points
        leaf_y, leaf_x = np.where(leaf_indices)

        while placed_px < target_lesion_px and attempts < 100:
            attempts += 1
            idx = random.randint(0, len(leaf_x) - 1)
            lx, ly = int(leaf_x[idx]), int(leaf_y[idx])
            rad = random.randint(8, max(9, int(np.sqrt(target_lesion_px / num_lesion_clusters) * 0.8)))

            temp_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(temp_mask, (lx, ly), rad, 255, -1)
            temp_mask = cv2.bitwise_and(temp_mask, gt_leaf_mask)

            gt_lesion_mask = cv2.bitwise_or(gt_lesion_mask, temp_mask)
            placed_px = int(np.count_nonzero(gt_lesion_mask == 255))

        # Color lesions in necrotic brown/dark orange (BGR: B=30, G=70, R=160)
        lesion_indices = (gt_lesion_mask == 255)
        leaf_layer[lesion_indices, 0] = np.random.randint(20, 45, np.count_nonzero(lesion_indices), dtype=np.uint8)
        leaf_layer[lesion_indices, 1] = np.random.randint(60, 95, np.count_nonzero(lesion_indices), dtype=np.uint8)
        leaf_layer[lesion_indices, 2] = np.random.randint(140, 190, np.count_nonzero(lesion_indices), dtype=np.uint8)

    # Composite leaf onto background
    image_bgr = bg.copy()
    image_bgr[leaf_indices] = leaf_layer[leaf_indices]

    # Compute exact ground-truth severity percentage
    actual_lesion_px = int(np.count_nonzero(gt_lesion_mask == 255))
    exact_sev = float((actual_lesion_px / max(leaf_area_px, 1)) * 100.0)

    return image_bgr, gt_leaf_mask, gt_lesion_mask, exact_sev


def split_by_plant_id(
    samples: List[BenchmarkSample],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[BenchmarkSample], List[BenchmarkSample], List[BenchmarkSample]]:
    """
    Performs strict Plant-ID level dataset splitting.
    Guarantees that all leaves originating from any individual plant ID are assigned
    strictly to the same partition, preventing identity leakage across evaluation splits.
    """
    random.seed(seed)
    # Group samples by plant_id
    plant_groups: Dict[str, List[BenchmarkSample]] = {}
    for s in samples:
        if s.plant_id not in plant_groups:
            plant_groups[s.plant_id] = []
        plant_groups[s.plant_id].append(s)

    unique_plants = list(plant_groups.keys())
    random.shuffle(unique_plants)

    n_total = len(unique_plants)
    n_train = max(1, int(n_total * train_ratio))
    n_val = max(1, int(n_total * val_ratio)) if (n_total - n_train) > 1 else 0

    train_plants = set(unique_plants[:n_train])
    val_plants = set(unique_plants[n_train:n_train + n_val])
    test_plants = set(unique_plants[n_train + n_val:])

    train_set = [s for pid in train_plants for s in plant_groups[pid]]
    val_set = [s for pid in val_plants for s in plant_groups[pid]]
    test_set = [s for pid in test_plants for s in plant_groups[pid]]

    return train_set, val_set, test_set


def create_synthetic_benchmark_dataset(
    num_plants: int = 10,
    leaves_per_plant: int = 4,
    seed: int = 42
) -> List[BenchmarkSample]:
    """
    Generates a multi-plant synthetic benchmark cohort with ground-truth labels.
    """
    samples: List[BenchmarkSample] = []
    idx = 0
    for p in range(num_plants):
        plant_id = f"PLANT_{p+1:03d}"
        for l in range(leaves_per_plant):
            leaf_id = f"LEAF_{l+1:02d}"
            target_sev = (idx % 7) * 6.5  # Range from 0% to ~39%
            img, m_leaf, m_lesion, sev = generate_synthetic_leaf(
                severity_target_pct=target_sev,
                seed=seed + idx
            )
            samples.append(BenchmarkSample(
                sample_id=f"{plant_id}_{leaf_id}",
                plant_id=plant_id,
                leaf_id=leaf_id,
                image_bgr=img,
                gt_leaf_mask=m_leaf,
                gt_lesion_mask=m_lesion,
                gt_severity_pct=sev,
                metadata={"synthetic": "true", "target_sev": str(target_sev)}
            ))
            idx += 1
    return samples
