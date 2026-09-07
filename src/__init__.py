"""
FieldSight-Lite: Illumination-Robust, Training-Free Plant Disease Severity
and Progression Monitoring on Edge Devices.
"""

from src.schemas import (
    ProcessingStatus,
    SeverityStatus,
    QualityStatus,
    ProgressionState,
    RiskAlertLevel,
    ProcessingError,
    QualityMetrics,
    SegmentationResult,
    SeverityResult,
    PerformanceMetrics,
    EvaluationMetrics,
    PerturbationResult,
    RobustnessReport,
    SeverityObservation,
    ProgressionReport,
    RuntimeMetadata,
    PipelineResult,
)
from src.pipeline import FieldSightPipeline
from src.quality import ImageQualityAssessor
from src.preprocessing import IlluminationNormalizer
from src.leaf_segmentation import LeafSegmenter
from src.lesion_segmentation import LesionSegmenter
from src.severity import SeverityCalculator
from src.robustness import RobustnessEngine
from src.progression import ProgressionLedger

__version__ = "1.0.0"
__all__ = [
    "FieldSightPipeline",
    "ImageQualityAssessor",
    "IlluminationNormalizer",
    "LeafSegmenter",
    "LesionSegmenter",
    "SeverityCalculator",
    "RobustnessEngine",
    "ProgressionLedger",
    "ProcessingStatus",
    "SeverityStatus",
    "QualityStatus",
    "ProgressionState",
    "RiskAlertLevel",
    "ProcessingError",
    "QualityMetrics",
    "SegmentationResult",
    "SeverityResult",
    "PerformanceMetrics",
    "EvaluationMetrics",
    "PerturbationResult",
    "RobustnessReport",
    "SeverityObservation",
    "ProgressionReport",
    "RuntimeMetadata",
    "PipelineResult",
]
