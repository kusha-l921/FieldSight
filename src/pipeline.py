"""
Master pipeline coordinator for FieldSight-Lite.
Orchestrates pre-flight quality verification, illumination normalization, two-pass segmentation,
severity quantification, active stress-testing, and temporal progression logging.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from src.logger import get_logger
from src.profiler import SystemProfiler
from src.provenance import get_runtime_metadata, load_config_dict
from src.quality import ImageQualityAssessor
from src.preprocessing import IlluminationNormalizer
from src.leaf_segmentation import LeafSegmenter
from src.lesion_segmentation import LesionSegmenter
from src.severity import SeverityCalculator
from src.robustness import RobustnessEngine
from src.progression import ProgressionLedger
from src.schemas import (
    PipelineResult,
    ProcessingError,
    ProcessingStatus,
    QualityMetrics,
    QualityStatus,
    RobustnessReport,
    SegmentationResult,
    SeverityResult,
    SeverityStatus,
)


def create_heatmap_overlay(
    original_bgr: np.ndarray,
    lesion_mask: np.ndarray,
    alpha: float = 0.45
) -> np.ndarray:
    """
    Overlays symptomatic lesion mask in translucent red over the original BGR image.
    Also outlines lesion boundaries for crisp visual localization.
    """
    if original_bgr is None:
        raise ValueError("original_bgr cannot be None")
    overlay = original_bgr.copy()
    if lesion_mask is None or np.count_nonzero(lesion_mask == 255) == 0:
        return overlay

    # Red color highlight in BGR: (0, 0, 255)
    red_color = np.array([0, 0, 240], dtype=np.uint8)
    mask_indices = (lesion_mask == 255)

    # Translucent blending on lesion area
    blended = cv2.addWeighted(original_bgr, 1.0 - alpha, np.full_like(original_bgr, red_color), alpha, 0.0)
    overlay[mask_indices] = blended[mask_indices]

    # Draw contour outlines
    contours, _ = cv2.findContours(lesion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 1)

    return overlay


class FieldSightPipeline:
    """
    Production-grade coordinator for the FieldSight-Lite computer vision architecture.
    Guarantees strict scientific integrity, zero-training inference, and deterministic outputs.
    """

    def __init__(self, config_dict_or_path: Union[str, Dict[str, Any]] = "configs/default_config.yaml"):
        if isinstance(config_dict_or_path, str):
            self.config = load_config_dict(config_dict_or_path)
            self.config_source = config_dict_or_path
        else:
            self.config = config_dict_or_path
            self.config_source = "dict"

        self.metadata = get_runtime_metadata(self.config)
        self.profiler = SystemProfiler(warmup_runs=0)

        # Initialize sub-modules
        self.quality_assessor = ImageQualityAssessor.from_config(self.config)
        self.normalizer = IlluminationNormalizer.from_config(self.config)
        self.leaf_segmenter = LeafSegmenter.from_config(self.config)
        self.lesion_segmenter = LesionSegmenter.from_config(self.config)
        self.severity_calculator = SeverityCalculator.from_config(self.config)
        self.robustness_engine = RobustnessEngine.from_config(self.config)
        self.progression_ledger = ProgressionLedger.from_config(self.config)

    def _internal_segment(self, bgr_img: np.ndarray) -> Tuple[SegmentationResult, SeverityResult]:
        """Helper for segmentation used during synthetic stress testing."""
        lab_norm, _, _ = self.normalizer.normalize(bgr_img)
        leaf_mask, leaf_area, _ = self.leaf_segmenter.segment_leaf(bgr_img, lab_image=lab_norm)
        seg_res = self.lesion_segmenter.segment_lesions(lab_norm, leaf_mask)
        q_dummy = self.quality_assessor.assess_quality(bgr_img)
        sev_res = self.severity_calculator.calculate_severity(seg_res, q_dummy, lab_norm)
        return seg_res, sev_res

    def process_image(
        self,
        image_input: Union[str, np.ndarray],
        plant_id: Optional[str] = None,
        leaf_id: Optional[str] = None,
        timestamp_utc: Optional[str] = None,
        run_stress_test: bool = False,
        execution_id: Optional[str] = None
    ) -> PipelineResult:
        """
        Executes full FieldSight-Lite pipeline on an input image.
        """
        t_start = time.perf_counter()
        logger = get_logger("fieldsight.pipeline", execution_id=execution_id)
        errors: List[ProcessingError] = []
        original_image_path: Optional[str] = None

        # 1. Load image
        if isinstance(image_input, str):
            original_image_path = image_input
            if not os.path.exists(image_input):
                err = ProcessingError(
                    code=ProcessingStatus.INVALID_INPUT,
                    message=f"Image file not found: {image_input}",
                    recoverable=False,
                    stage="IMAGE_LOAD"
                )
                logger.error(err.message)
                q_empty = self.quality_assessor.assess_quality(np.zeros((1, 1, 3), dtype=np.uint8))
                perf = self.profiler.get_performance_metrics(last_latency_ms=0.0)
                return PipelineResult(
                    status=ProcessingStatus.INVALID_INPUT,
                    errors=(err,),
                    metadata=self.metadata,
                    quality=q_empty,
                    segmentation=None,
                    severity=None,
                    performance=perf,
                    robustness=None,
                    progression=None,
                    overlay_image=None,
                    original_image_path=original_image_path
                )
            bgr_image = cv2.imread(image_input)
            if bgr_image is None:
                err = ProcessingError(
                    code=ProcessingStatus.INVALID_IMAGE,
                    message=f"Failed to decode image file: {image_input}",
                    recoverable=False,
                    stage="IMAGE_LOAD"
                )
                logger.error(err.message)
                q_empty = self.quality_assessor.assess_quality(np.zeros((1, 1, 3), dtype=np.uint8))
                perf = self.profiler.get_performance_metrics(last_latency_ms=0.0)
                return PipelineResult(
                    status=ProcessingStatus.INVALID_IMAGE,
                    errors=(err,),
                    metadata=self.metadata,
                    quality=q_empty,
                    segmentation=None,
                    severity=None,
                    performance=perf,
                    robustness=None,
                    progression=None,
                    overlay_image=None,
                    original_image_path=original_image_path
                )
        elif isinstance(image_input, np.ndarray):
            bgr_image = image_input
        else:
            err = ProcessingError(
                code=ProcessingStatus.INVALID_INPUT,
                message=f"Unsupported image input type: {type(image_input)}",
                recoverable=False,
                stage="IMAGE_LOAD"
            )
            q_empty = self.quality_assessor.assess_quality(np.zeros((1, 1, 3), dtype=np.uint8))
            perf = self.profiler.get_performance_metrics(last_latency_ms=0.0)
            return PipelineResult(
                status=ProcessingStatus.INVALID_INPUT,
                errors=(err,),
                metadata=self.metadata,
                quality=q_empty,
                segmentation=None,
                severity=None,
                performance=perf,
                robustness=None,
                progression=None,
                overlay_image=None,
                original_image_path=None
            )

        # 2. Pre-flight Quality Assessment
        with self.profiler.profile_stage("QUALITY_ASSESSMENT"):
            quality_metrics = self.quality_assessor.assess_quality(bgr_image)

        if not quality_metrics.is_valid_for_processing:
            # Map quality failure to typed processing error
            status_code = ProcessingStatus.PROCESSING_ERROR
            if quality_metrics.status == QualityStatus.BLURRY:
                status_code = ProcessingStatus.BLURRY_IMAGE
            elif quality_metrics.status == QualityStatus.DEGRADED_GLARE:
                status_code = ProcessingStatus.EXCELLENT if False else ProcessingStatus.EXCESSIVE_GLARE
            elif quality_metrics.status == QualityStatus.DEGRADED_SHADOW:
                status_code = ProcessingStatus.EXCESSIVE_SHADOW
            elif quality_metrics.status == QualityStatus.UNUSABLE:
                status_code = ProcessingStatus.INVALID_IMAGE

            err = ProcessingError(
                code=status_code,
                message=quality_metrics.diagnostic_message,
                recoverable=False,
                stage="QUALITY_ASSESSMENT"
            )
            logger.warning(f"Quality gate rejected frame: {quality_metrics.diagnostic_message}")
            t_end = time.perf_counter()
            perf = self.profiler.get_performance_metrics(last_latency_ms=(t_end - t_start) * 1000.0)

            return PipelineResult(
                status=status_code,
                errors=(err,),
                metadata=self.metadata,
                quality=quality_metrics,
                segmentation=None,
                severity=None,
                performance=perf,
                robustness=None,
                progression=None,
                overlay_image=None,
                original_image_path=original_image_path
            )

        # 3. Illumination Normalization
        with self.profiler.profile_stage("PREPROCESSING"):
            lab_norm, l_norm, bgr_norm = self.normalizer.normalize(bgr_image)

        # 4. Leaf Segmentation
        with self.profiler.profile_stage("LEAF_SEGMENTATION"):
            leaf_mask, leaf_area_px, is_valid_leaf = self.leaf_segmenter.segment_leaf(bgr_norm, lab_image=lab_norm)

        if not is_valid_leaf or leaf_area_px == 0:
            err = ProcessingError(
                code=ProcessingStatus.NO_LEAF,
                message=f"No valid leaf tissue segmented (area={leaf_area_px} px)",
                recoverable=False,
                stage="LEAF_SEGMENTATION"
            )
            logger.warning(err.message)
            t_end = time.perf_counter()
            perf = self.profiler.get_performance_metrics(last_latency_ms=(t_end - t_start) * 1000.0)

            return PipelineResult(
                status=ProcessingStatus.NO_LEAF,
                errors=(err,),
                metadata=self.metadata,
                quality=quality_metrics,
                segmentation=None,
                severity=None,
                performance=perf,
                robustness=None,
                progression=None,
                overlay_image=None,
                original_image_path=original_image_path
            )

        # Check for unsegmented full-canvas warning flag (> 0.98 total image area)
        h_orig, w_orig = bgr_image.shape[:2]
        if leaf_area_px > 0.98 * (h_orig * w_orig):
            warn = ProcessingError(
                code=ProcessingStatus.SEGMENTATION_FAILURE,
                message=f"Canopy mask spans {leaf_area_px}/{h_orig*w_orig} px (>98% of canvas), indicating potentially unsegmented background.",
                recoverable=True,
                stage="LEAF_SEGMENTATION"
            )
            errors.append(warn)
            logger.warning(warn.message)

        # 5. Lesion Segmentation
        with self.profiler.profile_stage("LESION_SEGMENTATION"):
            segmentation = self.lesion_segmenter.segment_lesions(lab_norm, leaf_mask)

        # 6. Active Stress Testing (Optional)
        robustness_report: Optional[RobustnessReport] = None
        lrs_value: Optional[float] = None
        if run_stress_test:
            with self.profiler.profile_stage("STRESS_TESTING"):
                initial_sev = self.severity_calculator.calculate_severity(
                    segmentation, quality_metrics, lab_norm
                )
                robustness_report = self.robustness_engine.evaluate_robustness(
                    original_bgr=bgr_image,
                    base_segmentation=segmentation,
                    base_severity=initial_sev,
                    segment_fn=self._internal_segment
                )
                lrs_value = robustness_report.lighting_robustness_score

        # 7. Severity & Confidence Calculation
        with self.profiler.profile_stage("SEVERITY_CALCULATION"):
            severity = self.severity_calculator.calculate_severity(
                segmentation=segmentation,
                quality=quality_metrics,
                lab_image=lab_norm,
                lrs_score=lrs_value
            )

        # 8. Visual Artifacts
        overlay_image = create_heatmap_overlay(bgr_image, segmentation.lesion_mask)

        # 9. Temporal Progression Tracking (Optional)
        progression_report: Optional[ProgressionReport] = None
        if plant_id is not None and leaf_id is not None:
            with self.profiler.profile_stage("PROGRESSION_LOGGING"):
                progression_report = self.progression_ledger.record_observation(
                    plant_id=plant_id,
                    leaf_id=leaf_id,
                    severity_pct=severity.severity_pct,
                    confidence_score=severity.confidence_score,
                    timestamp_utc=timestamp_utc,
                    raw_metadata={"quality_status": quality_metrics.status.value}
                )

        t_end = time.perf_counter()
        total_latency_ms = (t_end - t_start) * 1000.0
        perf = self.profiler.get_performance_metrics(last_latency_ms=total_latency_ms)

        logger.info(
            f"Analyzed leaf: severity={severity.severity_pct:.2f}%, confidence={severity.confidence_score:.1f}%, latency={total_latency_ms:.1f}ms",
            extra_data={"status": "SUCCESS", "lesion_px": segmentation.lesion_area_px}
        )

        return PipelineResult(
            status=ProcessingStatus.SUCCESS,
            errors=tuple(errors),
            metadata=self.metadata,
            quality=quality_metrics,
            segmentation=segmentation,
            severity=severity,
            performance=perf,
            robustness=robustness_report,
            progression=progression_report,
            overlay_image=overlay_image,
            original_image_path=original_image_path
        )

    def process_image_from_path(
        self,
        image_path: str,
        plant_id: Optional[str] = None,
        leaf_id: Optional[str] = None,
        timestamp_utc: Optional[str] = None,
        run_stress_test: bool = False,
        save_artifacts: bool = False,
        output_dir: Optional[str] = None,
        execution_id: Optional[str] = None
    ) -> PipelineResult:
        """
        Convenience wrapper executing process_image on a file path.
        Optionally saves visual overlay and JSON manifest if save_artifacts is True.
        """
        result = self.process_image(
            image_input=image_path,
            plant_id=plant_id,
            leaf_id=leaf_id,
            timestamp_utc=timestamp_utc,
            run_stress_test=run_stress_test,
            execution_id=execution_id
        )
        if save_artifacts and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            stem = os.path.splitext(os.path.basename(image_path))[0]
            if result.overlay_image is not None:
                cv2.imwrite(os.path.join(output_dir, f"{stem}_overlay.png"), result.overlay_image)
            self.export_json(result, os.path.join(output_dir, f"{stem}_manifest.json"))
        return result

    def export_result_dict(self, result: PipelineResult) -> Dict[str, Any]:
        """
        Converts a PipelineResult into the standardized JSON manifest specification.
        """
        manifest: Dict[str, Any] = {
            "status": result.status.value,
            "metadata": {
                "pipeline_version": result.metadata.pipeline_version,
                "config_hash": result.metadata.config_hash_sha256,
                "timestamp_utc": result.metadata.timestamp_utc
            },
            "quality": {
                "status": result.quality.status.value,
                "mean_lightness": round(result.quality.mean_lightness, 2),
                "glare_pixel_ratio": round(result.quality.glare_pixel_ratio, 4),
                "shadow_pixel_ratio": round(result.quality.shadow_pixel_ratio, 4),
                "laplacian_variance": round(result.quality.laplacian_variance, 2)
            }
        }

        if result.severity is not None:
            manifest["severity"] = {
                "severity_pct": round(result.severity.severity_pct, 2),
                "unaffected_pct": round(result.severity.unaffected_pct, 2),
                "confidence_score": round(result.severity.confidence_score, 1),
                "status": result.severity.status.value
            }

        manifest["performance"] = {
            "latency_mean_ms": round(result.performance.latency_mean_ms, 2),
            "latency_p95_ms": round(result.performance.latency_p95_ms, 2),
            "fps": round(result.performance.fps, 2)
        }

        if result.robustness is not None:
            manifest["robustness"] = {
                "lighting_robustness_score": round(result.robustness.lighting_robustness_score, 2),
                "mean_severity_drift": round(result.robustness.mean_severity_drift, 2)
            }

        if result.progression is not None:
            prog_rate = round(result.progression.progression_rate_daily, 2) if result.progression.progression_rate_daily is not None else None
            manifest["progression"] = {
                "progression_rate_daily": prog_rate,
                "state": result.progression.state.value,
                "risk_alert_level": result.progression.risk_alert_level.value
            }

        if result.errors:
            manifest["errors"] = [
                {"code": err.code.value, "message": err.message, "stage": err.stage}
                for err in result.errors
            ]

        return manifest

    def export_json(self, result: PipelineResult, output_path: str) -> None:
        """Saves result dictionary as a formatted JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        manifest = self.export_result_dict(result)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
