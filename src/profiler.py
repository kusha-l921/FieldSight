"""
Performance and resource profiler for FieldSight-Lite on edge devices.
Measures latency (mean, median, P95, FPS), psutil RAM footprint, and CPU utilization.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Dict, Generator, List, Optional
import numpy as np
import psutil

from src.schemas import PerformanceMetrics


class SystemProfiler:
    """
    Lightweight resource and latency profiler designed for continuous edge execution.
    Tracks stage durations, calculates statistical distributions, and queries hardware metrics.
    """

    def __init__(self, warmup_runs: int = 0):
        self.warmup_runs = warmup_runs
        self._latencies_ms: List[float] = []
        self._stage_latencies_ms: Dict[str, List[float]] = {}
        self._process = psutil.Process(os.getpid())
        self._initial_mem_bytes = self._get_rss_bytes()
        self._peak_mem_bytes = self._initial_mem_bytes

    def _get_rss_bytes(self) -> int:
        try:
            return self._process.memory_info().rss
        except Exception:
            return 0

    def record_run(self, latency_ms: float) -> None:
        """Records an end-to-end execution latency in milliseconds."""
        self._latencies_ms.append(float(latency_ms))
        current_mem = self._get_rss_bytes()
        if current_mem > self._peak_mem_bytes:
            self._peak_mem_bytes = current_mem

    def record_stage(self, stage_name: str, duration_ms: float) -> None:
        """Records latency for an isolated sub-stage."""
        if stage_name not in self._stage_latencies_ms:
            self._stage_latencies_ms[stage_name] = []
        self._stage_latencies_ms[stage_name].append(float(duration_ms))

    @contextmanager
    def profile_stage(self, stage_name: str) -> Generator[None, None, None]:
        """Context manager to measure execution time of a single pipeline step."""
        t_start = time.perf_counter()
        try:
            yield
        finally:
            t_end = time.perf_counter()
            duration_ms = (t_end - t_start) * 1000.0
            self.record_stage(stage_name, duration_ms)

    def get_performance_metrics(self, last_latency_ms: Optional[float] = None) -> PerformanceMetrics:
        """
        Computes summary statistics across accumulated runs or returns metrics for single run.
        """
        if last_latency_ms is not None:
            self.record_run(last_latency_ms)

        latencies = self._latencies_ms[self.warmup_runs:] if len(self._latencies_ms) > self.warmup_runs else self._latencies_ms
        if not latencies:
            if last_latency_ms is not None:
                latencies = [last_latency_ms]
            else:
                latencies = [0.0]

        lat_arr = np.array(latencies, dtype=np.float64)
        mean_ms = float(np.mean(lat_arr))
        median_ms = float(np.median(lat_arr))
        p95_ms = float(np.percentile(lat_arr, 95))
        fps = float(1000.0 / mean_ms) if mean_ms > 0.0 else 0.0

        peak_ram_mb: Optional[float] = None
        try:
            peak_ram_mb = float(self._peak_mem_bytes / (1024.0 * 1024.0))
        except Exception:
            peak_ram_mb = None

        cpu_util: Optional[float] = None
        try:
            cpu_util = float(self._process.cpu_percent(interval=None))
            cpu_util = max(0.0, min(100.0, cpu_util))
        except Exception:
            cpu_util = None

        return PerformanceMetrics(
            latency_mean_ms=mean_ms,
            latency_median_ms=median_ms,
            latency_p95_ms=p95_ms,
            fps=fps,
            peak_ram_mb=peak_ram_mb,
            cpu_utilization_pct=cpu_util
        )

    def get_stage_breakdown(self) -> Dict[str, float]:
        """Returns mean latency (ms) for each tracked sub-stage."""
        breakdown: Dict[str, float] = {}
        for stage, times in self._stage_latencies_ms.items():
            if times:
                breakdown[stage] = float(np.mean(times))
        return breakdown

    def reset(self) -> None:
        """Resets tracked latency and memory statistics."""
        self._latencies_ms.clear()
        self._stage_latencies_ms.clear()
        self._peak_mem_bytes = self._get_rss_bytes()
