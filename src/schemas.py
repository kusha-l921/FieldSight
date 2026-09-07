"""
Strict typed schemas and immutable data contracts for FieldSight-Lite.
Enforces runtime boundary checks, finite-float invariants, and read-only array protections.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, Tuple
import math
import numpy as np


def validate_finite(name: str, value: Optional[float]) -> None:
    """Validates that a numerical value is not None and is finite (not NaN or Inf)."""
    if value is not None and not math.isfinite(value):
        raise ValueError(f"Field '{name}' must be finite, got: {value}")


def make_array_readonly(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Returns a write-protected copy of the input numpy array."""
    if arr is not None and isinstance(arr, np.ndarray):
        arr_copy = arr.copy()
        arr_copy.setflags(write=False)
        return arr_copy
    return None


class ProcessingStatus(str, Enum):
    SUCCESS = "SUCCESS"
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
    NO_LEAF = "NO_LEAF"
    INVALID_IMAGE = "INVALID_IMAGE"
    BLURRY_IMAGE = "BLURRY_IMAGE"
    EXCESSIVE_GLARE = "EXCESSIVE_GLARE"
    EXCESSIVE_SHADOW = "EXCESSIVE_SHADOW"
    EMPTY_LESION = "EMPTY_LESION"
    CAMERA_ERROR = "CAMERA_ERROR"
    SEGMENTATION_FAILURE = "SEGMENTATION_FAILURE"
    MISSING_GROUND_TRUTH = "MISSING_GROUND_TRUTH"
    INVALID_GROUND_TRUTH = "INVALID_GROUND_TRUTH"
    INVALID_INPUT = "INVALID_INPUT"
    PROCESSING_ERROR = "PROCESSING_ERROR"


class SeverityStatus(str, Enum):
    NO_LESION_DETECTED = "NO_LESION_DETECTED"
    LESION_DETECTED = "LESION_DETECTED"
    INVALID_ZERO_LEAF = "INVALID_ZERO_LEAF"
    INVALID_SEGMENTATION = "INVALID_SEGMENTATION"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"


class QualityStatus(str, Enum):
    EXCELLENT = "EXCELLENT"
    ACCEPTABLE = "ACCEPTABLE"
    DEGRADED_GLARE = "DEGRADED_GLARE"
    DEGRADED_SHADOW = "DEGRADED_SHADOW"
    BLURRY = "BLURRY"
    UNUSABLE = "UNUSABLE"


class ProgressionState(str, Enum):
    STABLE = "STABLE"
    EMERGING = "EMERGING"
    MODERATE = "MODERATE"
    RAPID = "RAPID"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class RiskAlertLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    INDETERMINATE = "INDETERMINATE"


@dataclass(frozen=True)
class ProcessingError:
    code: ProcessingStatus
    message: str
    recoverable: bool
    stage: str


@dataclass(frozen=True)
class QualityMetrics:
    status: QualityStatus
    mean_lightness: float              # Physical CIELAB L* [0.0, 100.0]
    std_lightness: float               # Std dev of physical L* (>= 0.0)
    glare_pixel_ratio: float           # Specular ratio [0.0, 1.0]
    shadow_pixel_ratio: float          # Deep shadow ratio [0.0, 1.0]
    laplacian_variance: float          # Focus score (>= 0.0)
    is_valid_for_processing: bool
    diagnostic_message: str

    def __post_init__(self):
        validate_finite("mean_lightness", self.mean_lightness)
        validate_finite("std_lightness", self.std_lightness)
        validate_finite("glare_pixel_ratio", self.glare_pixel_ratio)
        validate_finite("shadow_pixel_ratio", self.shadow_pixel_ratio)
        validate_finite("laplacian_variance", self.laplacian_variance)
        if not (0.0 <= self.mean_lightness <= 100.0):
            raise ValueError(f"mean_lightness outside [0, 100]: {self.mean_lightness}")
        if self.std_lightness < 0.0:
            raise ValueError(f"std_lightness must be non-negative: {self.std_lightness}")
        if not (0.0 <= self.glare_pixel_ratio <= 1.0):
            raise ValueError(f"glare_pixel_ratio outside [0, 1]: {self.glare_pixel_ratio}")
        if not (0.0 <= self.shadow_pixel_ratio <= 1.0):
            raise ValueError(f"shadow_pixel_ratio outside [0, 1]: {self.shadow_pixel_ratio}")
        if self.laplacian_variance < 0.0:
            raise ValueError(f"laplacian_variance must be non-negative: {self.laplacian_variance}")


@dataclass(frozen=True)
class SegmentationResult:
    leaf_mask: np.ndarray              # uint8 array [H, W], values in {0, 255}
    lesion_mask: np.ndarray            # uint8 array [H, W], values in {0, 255}
    leaf_area_px: int                  # >= 0, equals np.count_nonzero(leaf_mask == 255)
    lesion_area_px: int                # >= 0, equals np.count_nonzero(lesion_mask == 255)
    healthy_mu_a: float                # Median a* of robust healthy inliers
    healthy_mad_a: float               # MAD of a* (>= 0.0)
    healthy_mu_b: float                # Median b* of robust healthy inliers
    healthy_mad_b: float               # MAD of b* (>= 0.0)

    def __post_init__(self):
        validate_finite("healthy_mu_a", self.healthy_mu_a)
        validate_finite("healthy_mad_a", self.healthy_mad_a)
        validate_finite("healthy_mu_b", self.healthy_mu_b)
        validate_finite("healthy_mad_b", self.healthy_mad_b)
        if self.leaf_mask is None or self.lesion_mask is None:
            raise ValueError("Masks cannot be None")
        if self.leaf_mask.dtype != np.uint8 or self.lesion_mask.dtype != np.uint8:
            raise TypeError("Masks must be uint8")
        if self.leaf_mask.shape != self.lesion_mask.shape:
            raise ValueError("Mask shapes must match")
        if len(self.leaf_mask.shape) != 2:
            raise ValueError("Masks must be 2D arrays")
        if not np.all(np.isin(self.leaf_mask, [0, 255])):
            raise ValueError("leaf_mask must contain only {0, 255}")
        if not np.all(np.isin(self.lesion_mask, [0, 255])):
            raise ValueError("lesion_mask must contain only {0, 255}")
        
        actual_leaf = int(np.count_nonzero(self.leaf_mask == 255))
        actual_lesion = int(np.count_nonzero(self.lesion_mask == 255))
        if self.leaf_area_px != actual_leaf:
            raise ValueError(f"leaf_area_px mismatch: {self.leaf_area_px} != {actual_leaf}")
        if self.lesion_area_px != actual_lesion:
            raise ValueError(f"lesion_area_px mismatch: {self.lesion_area_px} != {actual_lesion}")
        if self.healthy_mad_a < 0.0 or self.healthy_mad_b < 0.0:
            raise ValueError("MAD must be non-negative")
        if self.lesion_area_px > self.leaf_area_px:
            raise ValueError("Lesion area cannot exceed leaf area")
        
        invalid_px = int(np.count_nonzero((self.lesion_mask == 255) & (self.leaf_mask == 0)))
        if invalid_px > 0:
            raise ValueError(f"Invariant broken: {invalid_px} lesion pixels exist outside leaf mask")

        object.__setattr__(self, 'leaf_mask', make_array_readonly(self.leaf_mask))
        object.__setattr__(self, 'lesion_mask', make_array_readonly(self.lesion_mask))


@dataclass(frozen=True)
class SeverityResult:
    severity_pct: float                # [0.0, 100.0]
    unaffected_pct: float              # [0.0, 100.0]
    confidence_score: float            # Formally computed confidence [0.0, 100.0]
    status: SeverityStatus

    def __post_init__(self):
        validate_finite("severity_pct", self.severity_pct)
        validate_finite("unaffected_pct", self.unaffected_pct)
        validate_finite("confidence_score", self.confidence_score)
        if not (0.0 <= self.severity_pct <= 100.0):
            raise ValueError(f"severity_pct outside [0, 100]: {self.severity_pct}")
        if not (0.0 <= self.unaffected_pct <= 100.0):
            raise ValueError(f"unaffected_pct outside [0, 100]: {self.unaffected_pct}")
        if not (0.0 <= self.confidence_score <= 100.0):
            raise ValueError(f"confidence_score outside [0, 100]: {self.confidence_score}")
        if abs((self.severity_pct + self.unaffected_pct) - 100.0) > 1e-3:
            raise ValueError(f"severity_pct ({self.severity_pct}) and unaffected_pct ({self.unaffected_pct}) must sum to 100.0")


@dataclass(frozen=True)
class PerformanceMetrics:
    latency_mean_ms: float
    latency_median_ms: float
    latency_p95_ms: float
    fps: float
    peak_ram_mb: Optional[float]
    cpu_utilization_pct: Optional[float]

    def __post_init__(self):
        validate_finite("latency_mean_ms", self.latency_mean_ms)
        validate_finite("latency_median_ms", self.latency_median_ms)
        validate_finite("latency_p95_ms", self.latency_p95_ms)
        validate_finite("fps", self.fps)
        validate_finite("peak_ram_mb", self.peak_ram_mb)
        validate_finite("cpu_utilization_pct", self.cpu_utilization_pct)
        if self.latency_mean_ms < 0 or self.latency_median_ms < 0 or self.latency_p95_ms < 0:
            raise ValueError("Latency cannot be negative")
        if self.fps < 0:
            raise ValueError("FPS cannot be negative")
        if self.peak_ram_mb is not None and self.peak_ram_mb < 0:
            raise ValueError("RAM cannot be negative")
        if self.cpu_utilization_pct is not None and not (0.0 <= self.cpu_utilization_pct <= 100.0):
            raise ValueError(f"CPU % outside [0, 100]: {self.cpu_utilization_pct}")


@dataclass(frozen=True)
class EvaluationMetrics:
    leaf_iou: float
    leaf_dice: float
    lesion_iou: float
    lesion_dice: float
    lesion_precision: float
    lesion_recall: float
    severity_mae: float
    severity_rmse: float

    def __post_init__(self):
        for name, val in [("leaf_iou", self.leaf_iou), ("leaf_dice", self.leaf_dice),
                          ("lesion_iou", self.lesion_iou), ("lesion_dice", self.lesion_dice),
                          ("lesion_precision", self.lesion_precision), ("lesion_recall", self.lesion_recall)]:
            validate_finite(name, val)
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{name} outside [0.0, 1.0]: {val}")
        validate_finite("severity_mae", self.severity_mae)
        validate_finite("severity_rmse", self.severity_rmse)


@dataclass(frozen=True)
class PerturbationResult:
    perturbation_name: str
    perturbed_severity_pct: float
    severity_drift_delta: float
    leaf_mask_stability_iou: float
    lesion_mask_stability_iou: float

    def __post_init__(self):
        validate_finite("perturbed_severity_pct", self.perturbed_severity_pct)
        validate_finite("severity_drift_delta", self.severity_drift_delta)
        validate_finite("leaf_mask_stability_iou", self.leaf_mask_stability_iou)
        validate_finite("lesion_mask_stability_iou", self.lesion_mask_stability_iou)
        if not (0.0 <= self.perturbed_severity_pct <= 100.0):
            raise ValueError(f"perturbed_severity outside [0, 100]: {self.perturbed_severity_pct}")
        if self.severity_drift_delta < 0.0:
            raise ValueError(f"drift delta cannot be negative: {self.severity_drift_delta}")
        if not (0.0 <= self.leaf_mask_stability_iou <= 1.0):
            raise ValueError(f"leaf stability IoU outside [0, 1]: {self.leaf_mask_stability_iou}")
        if not (0.0 <= self.lesion_mask_stability_iou <= 1.0):
            raise ValueError(f"lesion stability IoU outside [0, 1]: {self.lesion_mask_stability_iou}")


@dataclass(frozen=True)
class RobustnessReport:
    original_severity: float
    mean_severity_drift: float
    severity_std_dev: float
    lighting_robustness_score: float   # LRS [0.0, 100.0]
    field_robustness_index: float      # FRI [0.0, 100.0]
    perturbation_breakdown: Tuple[PerturbationResult, ...]

    def __post_init__(self):
        validate_finite("original_severity", self.original_severity)
        validate_finite("mean_severity_drift", self.mean_severity_drift)
        validate_finite("severity_std_dev", self.severity_std_dev)
        validate_finite("lighting_robustness_score", self.lighting_robustness_score)
        validate_finite("field_robustness_index", self.field_robustness_index)
        if not (0.0 <= self.lighting_robustness_score <= 100.0):
            raise ValueError(f"LRS outside [0, 100]: {self.lighting_robustness_score}")
        if not (0.0 <= self.field_robustness_index <= 100.0):
            raise ValueError(f"FRI outside [0, 100]: {self.field_robustness_index}")


@dataclass(frozen=True)
class SeverityObservation:
    plant_id: str
    leaf_id: str
    timestamp_utc: str
    severity_pct: float
    confidence_score: float

    def __post_init__(self):
        validate_finite("severity_pct", self.severity_pct)
        validate_finite("confidence_score", self.confidence_score)
        if not (0.0 <= self.severity_pct <= 100.0):
            raise ValueError(f"severity_pct outside [0, 100]: {self.severity_pct}")
        if not (0.0 <= self.confidence_score <= 100.0):
            raise ValueError(f"confidence_score outside [0, 100]: {self.confidence_score}")


@dataclass(frozen=True)
class ProgressionReport:
    plant_id: str
    leaf_id: str
    current_timestamp_utc: str
    previous_timestamp_utc: Optional[str]
    delta_days: float
    delta_severity: float
    progression_rate_daily: Optional[float]
    state: ProgressionState
    risk_alert_level: RiskAlertLevel
    observation_history: Tuple[SeverityObservation, ...]

    def __post_init__(self):
        validate_finite("delta_days", self.delta_days)
        validate_finite("delta_severity", self.delta_severity)
        validate_finite("progression_rate_daily", self.progression_rate_daily)
        if self.delta_days < 0.0:
            raise ValueError(f"delta_days cannot be negative: {self.delta_days}")


@dataclass(frozen=True)
class RuntimeMetadata:
    python_version: str
    opencv_version: str
    numpy_version: str
    platform_system: str
    cpu_architecture: str
    timestamp_utc: str
    pipeline_version: str
    config_hash_sha256: str


@dataclass(frozen=True)
class PipelineResult:
    status: ProcessingStatus
    errors: Tuple[ProcessingError, ...]
    metadata: RuntimeMetadata
    quality: QualityMetrics
    segmentation: Optional[SegmentationResult]
    severity: Optional[SeverityResult]
    performance: PerformanceMetrics
    robustness: Optional[RobustnessReport]
    progression: Optional[ProgressionReport]
    overlay_image: Optional[np.ndarray]
    original_image_path: Optional[str]

    def __post_init__(self):
        if self.overlay_image is not None:
            if self.overlay_image.dtype != np.uint8:
                raise TypeError("overlay_image must be uint8")
            if len(self.overlay_image.shape) != 3 or self.overlay_image.shape[2] != 3:
                raise ValueError(f"overlay_image must have shape (H, W, 3), got {self.overlay_image.shape}")
            object.__setattr__(self, 'overlay_image', make_array_readonly(self.overlay_image))
