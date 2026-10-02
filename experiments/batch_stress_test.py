"""
Real-Leaf Batch Stress-Testing & Empirical Robustness Evaluation.
Ingests real foliage imagery from data/raw/Tomato_Early_blight/, executes the full
FieldSight-Lite pipeline with active 5-perturbation stress testing, compiles per-leaf
empirical metrics into a structured CSV, and presents publication-grade summary statistics.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
from rich.table import Table

# Ensure workspace root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import FieldSightPipeline
from src.schemas import ProcessingStatus


# Mapping from standard perturbation names to clean column names
PERTURBATION_MAP = {
    "Shadow Ramp": "drift_shadow_ramp",
    "Specular Glare": "drift_specular_glare",
    "Overexposure (+55)": "drift_overexposure",
    "Underexposure (0.5x)": "drift_underexposure",
    "Gamma Shift (γ=1.8)": "drift_gamma_shift",
}


def find_leaf_images(input_dir: str, max_candidates: int = 200) -> List[str]:
    """
    Finds real leaf image files in input directory sorted deterministically.
    Supports .JPG, .jpg, .jpeg, .png, .PNG extensions.
    """
    valid_exts = {".jpg", ".jpeg", ".png", ".JPG", ".PNG", ".JPEG"}
    all_files = []
    for entry in sorted(os.listdir(input_dir)):
        ext = os.path.splitext(entry)[1]
        if ext in valid_exts:
            full_path = os.path.join(input_dir, entry)
            if os.path.isfile(full_path):
                all_files.append(full_path)
        if len(all_files) >= max_candidates:
            break

    return all_files


def run_batch_stress_test(
    candidate_paths: List[str],
    pipeline: FieldSightPipeline,
    console: Console,
    target_count: int = 100
) -> pd.DataFrame:
    """
    Executes pipeline with active stress testing over real leaf imagery.
    Continues until target_count successfully processed leaves are collected.
    Aggregates baseline severity, LRS, MASD, standard deviation, FRI, latency,
    and individual perturbation drift deltas.
    """
    records: List[Dict[str, Any]] = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("• ({task.completed}/{task.total} evaluated)"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("[green]Evaluating real leaves...", total=target_count)

        for img_path in candidate_paths:
            if len(records) >= target_count:
                break

            fname = os.path.basename(img_path)
            progress.update(task_id, description=f"[cyan]Processing {fname[:24]}...")

            t_start = time.perf_counter()
            result = pipeline.process_image_from_path(
                image_path=img_path,
                run_stress_test=True,
                save_artifacts=False
            )
            t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0

            if result.status != ProcessingStatus.SUCCESS or result.severity is None or result.robustness is None:
                console.print(f"[yellow]Quality Filter:[/] Excluded frame {fname} ({result.status.value})")
                continue

            rob = result.robustness
            sev = result.severity
            perf = result.performance

            row: Dict[str, Any] = {
                "filename": fname,
                "baseline_severity_pct": round(sev.severity_pct, 3),
                "lighting_robustness_score": round(rob.lighting_robustness_score, 3),
                "mean_severity_drift": round(rob.mean_severity_drift, 3),
                "severity_std_dev": round(rob.severity_std_dev, 3),
                "field_robustness_index": round(rob.field_robustness_index, 3),
                "latency_mean_ms": round(perf.latency_mean_ms if perf.latency_mean_ms > 0 else t_elapsed_ms, 2),
            }

            # Extract individual drift deltas for each of the 5 perturbations
            pert_deltas: Dict[str, float] = {}
            for p in rob.perturbation_breakdown:
                for std_name, col_name in PERTURBATION_MAP.items():
                    if std_name in p.perturbation_name:
                        pert_deltas[col_name] = round(p.severity_drift_delta, 3)
                        break

            # Populate mapped columns with default 0.0 if not found
            for col_name in PERTURBATION_MAP.values():
                row[col_name] = pert_deltas.get(col_name, 0.0)

            records.append(row)
            progress.advance(task_id)

    return pd.DataFrame(records)


def print_console_summary(df: pd.DataFrame, console: Console) -> None:
    """
    Displays formatted publication-grade summary statistics for evaluated metrics.
    """
    n = len(df)
    if n == 0:
        console.print("[red]No leaves were successfully processed.[/]")
        return

    # Calculate statistics
    sev_mean, sev_std = df["baseline_severity_pct"].mean(), df["baseline_severity_pct"].std()
    lrs_mean, lrs_std = df["lighting_robustness_score"].mean(), df["lighting_robustness_score"].std()
    masd_mean, masd_std = df["mean_severity_drift"].mean(), df["mean_severity_drift"].std()
    fri_mean, fri_std = df["field_robustness_index"].mean(), df["field_robustness_index"].std()
    lat_mean, lat_std = df["latency_mean_ms"].mean(), df["latency_mean_ms"].std()
    effective_fps = 1000.0 / lat_mean if lat_mean > 0 else 0.0

    table = Table(
        title=f"FieldSight-Lite Empirical Stress-Testing Summary (N={n} Real Leaves)",
        show_header=True,
        header_style="bold magenta",
        title_justify="center"
    )
    table.add_column("Metric", style="bold cyan", justify="left")
    table.add_column("Symbol", justify="center", style="yellow")
    table.add_column("Mean ± Std Dev", justify="right", style="green")
    table.add_column("Median [IQR]", justify="right", style="white")
    table.add_column("Min – Max", justify="right", style="blue")

    def format_row(name: str, symbol: str, series: pd.Series, unit: str = "%") -> List[str]:
        m = series.mean()
        s = series.std()
        med = series.median()
        q25, q75 = series.quantile(0.25), series.quantile(0.75)
        mn, mx = series.min(), series.max()
        return [
            name,
            symbol,
            f"{m:.2f} ± {s:.2f}{unit}",
            f"{med:.2f} [{q25:.2f} – {q75:.2f}]{unit}",
            f"{mn:.2f} – {mx:.2f}{unit}"
        ]

    table.add_row(*format_row("Baseline Severity", "S_0", df["baseline_severity_pct"], "%"))
    table.add_row(*format_row("Lighting Robustness Score", "LRS", df["lighting_robustness_score"], "%"))
    table.add_row(*format_row("Mean Absolute Severity Drift", "MASD", df["mean_severity_drift"], "%"))
    table.add_row(*format_row("Severity Std Deviation", "σ_S", df["severity_std_dev"], "%"))
    table.add_row(*format_row("Field Robustness Index", "FRI", df["field_robustness_index"], "%"))
    table.add_row(*format_row("Processing Latency", "T_lat", df["latency_mean_ms"], " ms"))

    console.print("")
    console.print(table)
    console.print(
        Panel.fit(
            f"[bold green]Throughput:[/] [cyan]{effective_fps:.2f} FPS[/] (Effective Real-Time Edge Throughput)\n"
            f"[bold green]Dataset Source:[/] Tomato Early Blight (Real Foliage Cohort)\n"
            f"[bold green]Stress Suite:[/] 5 Deterministic Perturbations per Image",
            title="[bold yellow]Edge Performance Characteristics[/]"
        )
    )

    # Per-perturbation breakdown table
    p_table = Table(
        title="Breakdown by Environmental Perturbation (Severity Drift ΔS)",
        show_header=True,
        header_style="bold blue"
    )
    p_table.add_column("Perturbation", style="bold cyan")
    p_table.add_column("Mean Drift ΔS (%)", justify="right", style="magenta")
    p_table.add_column("Std Dev (%)", justify="right", style="white")
    p_table.add_column("Max Drift (%)", justify="right", style="red")

    for std_name, col_name in PERTURBATION_MAP.items():
        if col_name in df.columns:
            s = df[col_name]
            p_table.add_row(
                std_name,
                f"{s.mean():.2f}%",
                f"{s.std():.2f}%",
                f"{s.max():.2f}%"
            )

    console.print("")
    console.print(p_table)
    console.print("")


def main():
    parser = argparse.ArgumentParser(description="FieldSight-Lite Real-Leaf Batch Stress-Test Harness")
    parser.add_argument(
        "--input-dir",
        default="data/raw/Tomato_Early_blight",
        help="Directory containing real leaf imagery"
    )
    parser.add_argument(
        "--config",
        default="configs/default_config.yaml",
        help="Path to pipeline configuration YAML"
    )
    parser.add_argument(
        "--output-csv",
        default="data/benchmark_results/real_stress_test_metrics.csv",
        help="Destination CSV path for compiled metrics"
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=100,
        help="Number of real leaf images to evaluate (default: 100)"
    )
    args = parser.parse_args()

    console = Console()
    console.print(
        Panel.fit(
            "[bold green]FieldSight-Lite[/]: Empirical Real-Leaf Lighting Stress-Testing Harness\n"
            f"Input Directory: [cyan]{args.input_dir}[/]\n"
            f"Target Sample Count: [yellow]{args.num_images}[/] leaves\n"
            f"Output Destination: [magenta]{args.output_csv}[/]",
            title="[bold blue]Empirical Benchmarking Protocol[/]"
        )
    )

    if not os.path.exists(args.input_dir):
        console.print(f"[bold red]Error:[/] Input directory '{args.input_dir}' does not exist.")
        sys.exit(1)

    image_paths = find_leaf_images(args.input_dir, max_candidates=max(args.num_images * 2, 200))
    if not image_paths:
        console.print(f"[bold red]Error:[/] No supported leaf images found in '{args.input_dir}'.")
        sys.exit(1)

    console.print(f"[green]Discovered [bold]{len(image_paths)}[/] candidate real leaf images for stress evaluation.[/]\n")

    # Initialize master pipeline
    pipeline = FieldSightPipeline(args.config)

    # Execute batch test
    df_results = run_batch_stress_test(image_paths, pipeline, console, target_count=args.num_images)

    # Save to CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    df_results.to_csv(args.output_csv, index=False)
    console.print(f"[bold green]✓[/] Successfully compiled {len(df_results)} leaf records to: [cyan]{args.output_csv}[/]")

    # Output formatted console statistics
    print_console_summary(df_results, console)


if __name__ == "__main__":
    main()
