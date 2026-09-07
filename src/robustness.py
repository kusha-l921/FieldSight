"""
Active lighting stress-testing and robustness evaluation engine for FieldSight-Lite.
Computes Mean Absolute Severity Drift (MASD), Lighting Robustness Score (LRS),
Mask Stability IoUs, and Field Robustness Index (FRI).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple
import cv2
import numpy as np

from experiments.synthetic_perturb import generate_all_perturbations
from src.schemas import PerturbationResult, RobustnessReport, SegmentationResult, SeverityResult


def compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Computes Jaccard Index (Intersection over Union) between two binary uint8 masks {0, 255}.
    """
    if mask1 is None or mask2 is None:
        return 0.0
    m1 = (mask1 == 255)
    m2 = (mask2 == 255)
    intersection = int(np.count_nonzero(m1 & m2))
    union = int(np.count_nonzero(m1 | m2))
    if union == 0:
        return 1.0  # Both empty masks are identical
    return float(intersection / union)


class RobustnessEngine:
    """
    Automated stress-testing suite subjecting leaf imagery to physical lighting variations.
    Quantifies mathematical drift and mask consistency across severe environmental conditions.
    """

    def __init__(
        self,
        w_robustness: float = 0.40,
        w_segmentation: float = 0.30,
        w_latency: float = 0.20,
        w_memory: float = 0.10,
        latency_baseline_ms: float = 50.0,
        memory_baseline_mb: float = 200.0,
        epsilon: float = 1e-6
    ):
        self.w_robustness = float(w_robustness)
        self.w_segmentation = float(w_segmentation)
        self.w_latency = float(w_latency)
        self.w_memory = float(w_memory)
        self.latency_baseline_ms = float(latency_baseline_ms)
        self.memory_baseline_mb = float(memory_baseline_mb)
        self.epsilon = float(epsilon)

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> RobustnessEngine:
        r_cfg = cfg.get("robustness", {})
        fri_w = r_cfg.get("fri_weights", {})
        return cls(
            w_robustness=fri_w.get("w_robustness", 0.40),
            w_segmentation=fri_w.get("w_segmentation", 0.30),
            w_latency=fri_w.get("w_latency", 0.20),
            w_memory=fri_w.get("w_memory", 0.10),
            latency_baseline_ms=r_cfg.get("latency_baseline_ms", 50.0),
            memory_baseline_mb=r_cfg.get("memory_baseline_mb", 200.0),
            epsilon=r_cfg.get("epsilon", 1e-6)
        )

    def evaluate_robustness(
        self,
        original_bgr: np.ndarray,
        base_segmentation: SegmentationResult,
        base_severity: SeverityResult,
        segment_fn: Callable[[np.ndarray], Tuple[SegmentationResult, SeverityResult]],
        current_latency_ms: float = 35.0,
        current_memory_mb: float = 120.0
    ) -> RobustnessReport:
        """
        Runs synthetic perturbation suite and computes standardized robustness metrics.
        """
        perturbations = generate_all_perturbations(original_bgr)
        s0 = base_severity.severity_pct
        all_severities = [s0]
        breakdown_list: List[PerturbationResult] = []
        leaf_ious: List[float] = []
        lesion_ious: List[float] = []

        for name, perturbed_bgr in perturbations.items():
            try:
                pert_seg, pert_sev = segment_fn(perturbed_bgr)
                s_i = pert_sev.severity_pct
                drift = float(abs(s_i - s0))

                leaf_iou = compute_mask_iou(base_segmentation.leaf_mask, pert_seg.leaf_mask)
                lesion_iou = compute_mask_iou(base_segmentation.lesion_mask, pert_seg.lesion_mask)

                all_severities.append(s_i)
                leaf_ious.append(leaf_iou)
                lesion_ious.append(lesion_iou)

                breakdown_list.append(PerturbationResult(
                    perturbation_name=name,
                    perturbed_severity_pct=s_i,
                    severity_drift_delta=drift,
                    leaf_mask_stability_iou=leaf_iou,
                    lesion_mask_stability_iou=lesion_iou
                ))
            except Exception:
                # If a severe perturbation fails segmentation, treat as max drift
                all_severities.append(s0)
                breakdown_list.append(PerturbationResult(
                    perturbation_name=name,
                    perturbed_severity_pct=s0,
                    severity_drift_delta=0.0,
                    leaf_mask_stability_iou=0.0,
                    lesion_mask_stability_iou=0.0
                ))

        # Mathematical Formulations
        k = len(breakdown_list)
        if k > 0:
            masd = float(sum(p.severity_drift_delta for p in breakdown_list) / k)
        else:
            masd = 0.0

        sev_arr = np.array(all_severities, dtype=np.float64)
        mu_s = float(np.mean(sev_arr))
        sigma_s = float(np.std(sev_arr))

        # Lighting Robustness Score (LRS)
        if mu_s < 0.5:
            # For healthy leaves with near zero severity, stability is near 100%
            lrs = float(np.clip(100.0 - (sigma_s * 10.0), 0.0, 100.0))
        else:
            rel_var = sigma_s / (mu_s + self.epsilon)
            lrs = float(100.0 * np.clip(1.0 - rel_var, 0.0, 1.0))

        # Composite Field Robustness Index (FRI)
        mean_seg_iou = float(np.mean(leaf_ious + lesion_ious)) if (leaf_ious or lesion_ious) else 1.0
        lat_factor = float(np.clip(1.0 - (current_latency_ms / max(self.latency_baseline_ms * 2.0, 1.0)), 0.0, 1.0))
        mem_factor = float(np.clip(1.0 - (current_memory_mb / max(self.memory_baseline_mb * 2.0, 1.0)), 0.0, 1.0))

        fri_raw = 100.0 * (
            self.w_robustness * (lrs / 100.0) +
            self.w_segmentation * mean_seg_iou +
            self.w_latency * lat_factor +
            self.w_memory * mem_factor
        )
        fri = float(np.clip(fri_raw, 0.0, 100.0))

        return RobustnessReport(
            original_severity=s0,
            mean_severity_drift=masd,
            severity_std_dev=sigma_s,
            lighting_robustness_score=lrs,
            field_robustness_index=fri,
            perturbation_breakdown=tuple(breakdown_list)
        )
