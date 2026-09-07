"""
Evaluation metrics engine for FieldSight-Lite benchmarking.
Calculates IoU, Dice Coefficient, Precision, Recall, MAE, and RMSE for leaf & lesion segmentation.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple
import numpy as np

from src.schemas import EvaluationMetrics


def compute_iou(mask_pred: np.ndarray, mask_gt: np.ndarray) -> float:
    """Computes Jaccard Index / IoU between binary uint8 masks {0, 255}."""
    if mask_pred is None or mask_gt is None:
        return 0.0
    p = (mask_pred == 255)
    g = (mask_gt == 255)
    intersection = int(np.count_nonzero(p & g))
    union = int(np.count_nonzero(p | g))
    if union == 0:
        return 1.0  # Perfect match on empty ground truth
    return float(intersection / union)


def compute_dice(mask_pred: np.ndarray, mask_gt: np.ndarray) -> float:
    """Computes Dice Similarity Coefficient (F1-score of mask)."""
    if mask_pred is None or mask_gt is None:
        return 0.0
    p = (mask_pred == 255)
    g = (mask_gt == 255)
    intersection = int(np.count_nonzero(p & g))
    total_area = int(np.count_nonzero(p) + np.count_nonzero(g))
    if total_area == 0:
        return 1.0
    return float((2.0 * intersection) / total_area)


def compute_precision_recall(mask_pred: np.ndarray, mask_gt: np.ndarray) -> Tuple[float, float]:
    """Computes voxel-level Precision and Recall."""
    if mask_pred is None or mask_gt is None:
        return 0.0, 0.0
    p = (mask_pred == 255)
    g = (mask_gt == 255)
    tp = int(np.count_nonzero(p & g))
    fp = int(np.count_nonzero(p & ~g))
    fn = int(np.count_nonzero(~p & g))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else (1.0 if fn == 0 else 0.0)
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else (1.0 if fp == 0 else 0.0)
    return precision, recall


def evaluate_predictions(
    pred_leaf_masks: List[np.ndarray],
    pred_lesion_masks: List[np.ndarray],
    pred_severities: List[float],
    gt_leaf_masks: List[np.ndarray],
    gt_lesion_masks: List[np.ndarray],
    gt_severities: List[float]
) -> EvaluationMetrics:
    """
    Computes dataset-level aggregate performance metrics.
    """
    n = len(gt_severities)
    if n == 0:
        return EvaluationMetrics(
            leaf_iou=1.0, leaf_dice=1.0,
            lesion_iou=1.0, lesion_dice=1.0,
            lesion_precision=1.0, lesion_recall=1.0,
            severity_mae=0.0, severity_rmse=0.0
        )

    leaf_ious: List[float] = []
    leaf_dices: List[float] = []
    lesion_ious: List[float] = []
    lesion_dices: List[float] = []
    precisions: List[float] = []
    recalls: List[float] = []
    sev_diffs: List[float] = []

    for i in range(n):
        l_iou = compute_iou(pred_leaf_masks[i], gt_leaf_masks[i])
        l_dice = compute_dice(pred_leaf_masks[i], gt_leaf_masks[i])
        les_iou = compute_iou(pred_lesion_masks[i], gt_lesion_masks[i])
        les_dice = compute_dice(pred_lesion_masks[i], gt_lesion_masks[i])
        prec, rec = compute_precision_recall(pred_lesion_masks[i], gt_lesion_masks[i])

        leaf_ious.append(l_iou)
        leaf_dices.append(l_dice)
        lesion_ious.append(les_iou)
        lesion_dices.append(les_dice)
        precisions.append(prec)
        recalls.append(rec)

        sev_diffs.append(pred_severities[i] - gt_severities[i])

    mae = float(np.mean(np.abs(sev_diffs)))
    rmse = float(np.sqrt(np.mean(np.array(sev_diffs) ** 2)))

    return EvaluationMetrics(
        leaf_iou=float(np.mean(leaf_ious)),
        leaf_dice=float(np.mean(leaf_dices)),
        lesion_iou=float(np.mean(lesion_ious)),
        lesion_dice=float(np.mean(lesion_dices)),
        lesion_precision=float(np.mean(precisions)),
        lesion_recall=float(np.mean(recalls)),
        severity_mae=mae,
        severity_rmse=rmse
    )
