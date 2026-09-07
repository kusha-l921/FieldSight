"""
Provenance tracking and runtime environment metadata generator for FieldSight-Lite.
Provides deterministic SHA-256 configuration hashing and hardware/platform metadata capture.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional
import yaml
import cv2
import numpy as np

from src.schemas import RuntimeMetadata


def load_config_dict(config_path: str) -> Dict[str, Any]:
    """Loads a YAML configuration file into a dictionary."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML config format in {config_path}, expected dictionary")
    return cfg


def compute_config_hash(config_dict_or_path: Any) -> str:
    """
    Computes a deterministic SHA-256 hash of the configuration dictionary or file.
    Normalizes dictionary keys recursively to guarantee deterministic hash outputs.
    """
    if isinstance(config_dict_or_path, str):
        config_dict = load_config_dict(config_dict_or_path)
    elif isinstance(config_dict_or_path, dict):
        config_dict = config_dict_or_path
    else:
        raise TypeError(f"Expected file path or dict, got {type(config_dict_or_path)}")

    # Deterministic JSON serialization
    serialized = json.dumps(config_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def get_runtime_metadata(
    config_dict_or_path: Any,
    pipeline_version: Optional[str] = None
) -> RuntimeMetadata:
    """
    Constructs an immutable RuntimeMetadata record capturing all system, dependency,
    and configuration provenance at execution time.
    """
    if isinstance(config_dict_or_path, str):
        cfg = load_config_dict(config_dict_or_path)
    else:
        cfg = config_dict_or_path

    version = pipeline_version or str(cfg.get("version", "1.0.0"))
    cfg_hash = compute_config_hash(cfg)
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return RuntimeMetadata(
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        opencv_version=str(cv2.__version__),
        numpy_version=str(np.__version__),
        platform_system=str(platform.system()),
        cpu_architecture=str(platform.machine()),
        timestamp_utc=now_utc,
        pipeline_version=version,
        config_hash_sha256=cfg_hash
    )
