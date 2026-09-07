"""
Unified Click CLI for FieldSight-Lite platform.
Supports single image analysis, batch directory processing, live camera inference,
experimental benchmarking, and temporal progression querying.
"""

from __future__ import annotations

import glob
import os
import sys
from typing import Optional

# Add workspace directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import click
import cv2
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from experiments.dataset_loader import (
    create_synthetic_benchmark_dataset,
    split_by_plant_id,
)
from experiments.run_benchmarks import export_latex_table, run_benchmark_suite
from src.camera import CameraCapture
from src.pipeline import FieldSightPipeline
from src.progression import ProgressionLedger

console = Console()


@click.group()
@click.version_option("1.0.0", prog_name="FieldSight-Lite")
def cli():
    """FieldSight-Lite: Illumination-Robust Plant Disease Severity Platform."""
    pass


@cli.command("image")
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True), help="Path to input leaf image")
@click.option("--config", "-c", "config_path", default="configs/default_config.yaml", help="Path to YAML configuration")
@click.option("--output-dir", "-o", "output_dir", default="outputs", help="Directory to save visual artifacts and JSON")
@click.option("--stress-test", is_flag=True, default=False, help="Run active 5-perturbation lighting stress test")
@click.option("--plant-id", default=None, help="Optional plant identifier for progression ledger")
@click.option("--leaf-id", default=None, help="Optional leaf identifier for progression ledger")
def analyze_image_cmd(
    input_path: str,
    config_path: str,
    output_dir: str,
    stress_test: bool,
    plant_id: Optional[str],
    leaf_id: Optional[str]
):
    """Analyze a single leaf image and export structured diagnostics."""
    console.print(Panel.fit("[bold green]FieldSight-Lite: Single Image Inspection[/bold green]"))

    pipeline = FieldSightPipeline(config_path)
    res = pipeline.process_image(
        image_input=input_path,
        plant_id=plant_id,
        leaf_id=leaf_id,
        run_stress_test=stress_test
    )

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    json_path = os.path.join(output_dir, f"{base_name}_result.json")
    pipeline.export_json(res, json_path)

    # Save visual artifacts if successful
    if res.segmentation is not None:
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_leaf_mask.png"), res.segmentation.leaf_mask)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_lesion_mask.png"), res.segmentation.lesion_mask)
    if res.overlay_image is not None:
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_heatmap_overlay.png"), res.overlay_image)

    # Print summary table
    table = Table(title=f"Analysis Summary for {os.path.basename(input_path)}")
    table.add_column("Metric / Field", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    table.add_row("Status", str(res.status.value))
    table.add_row("Quality Status", str(res.quality.status.value))
    table.add_row("Mean Lightness (L*)", f"{res.quality.mean_lightness:.1f}")
    table.add_row("Laplacian Var (Focus)", f"{res.quality.laplacian_variance:.1f}")

    if res.severity is not None:
        table.add_row("Disease Severity (%)", f"[bold red]{res.severity.severity_pct:.2f}%[/bold red]")
        table.add_row("Unaffected (%)", f"{res.severity.unaffected_pct:.2f}%")
        table.add_row("Confidence Score", f"{res.severity.confidence_score:.1f}%")
        table.add_row("Severity Status", str(res.severity.status.value))

    table.add_row("Latency (Mean)", f"{res.performance.latency_mean_ms:.1f} ms")
    table.add_row("FPS", f"{res.performance.fps:.1f}")

    if res.robustness is not None:
        table.add_row("Lighting Robustness (LRS)", f"{res.robustness.lighting_robustness_score:.1f}%")
        table.add_row("Severity Drift (MASD)", f"{res.robustness.mean_severity_drift:.2f}%")
        table.add_row("Field Robustness (FRI)", f"{res.robustness.field_robustness_index:.1f}%")

    if res.progression is not None:
        rate_str = f"{res.progression.progression_rate_daily:.2f} %/day" if res.progression.progression_rate_daily is not None else "N/A"
        table.add_row("Progression Rate", rate_str)
        table.add_row("Progression State", str(res.progression.state.value))
        table.add_row("Risk Alert Level", str(res.progression.risk_alert_level.value))

    console.print(table)
    console.print(f"[bold green]✓ Visual artifacts and JSON saved to:[/bold green] {output_dir}")


@cli.command("batch")
@click.option("--input-dir", "-i", "input_dir", required=True, type=click.Path(exists=True), help="Directory containing images")
@click.option("--output-dir", "-o", "output_dir", default="batch_outputs", help="Directory to save output files")
@click.option("--config", "-c", "config_path", default="configs/default_config.yaml", help="Configuration file")
@click.option("--export-csv", is_flag=True, default=True, help="Export batch CSV summary")
def batch_analysis_cmd(input_dir: str, output_dir: str, config_path: str, export_csv: bool):
    """Batch process a directory of foliage images."""
    console.print(Panel.fit("[bold green]FieldSight-Lite: Batch Image Pipeline[/bold green]"))

    exts = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")
    image_paths = []
    for ext in exts:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    image_paths = sorted(list(set(image_paths)))

    if not image_paths:
        console.print(f"[red]No valid image files found in {input_dir}[/red]")
        return

    console.print(f"Discovered [bold cyan]{len(image_paths)}[/bold cyan] images. Initializing pipeline...")
    pipeline = FieldSightPipeline(config_path)
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    with click.progressbar(image_paths, label="Processing images") as bar:
        for img_path in bar:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            res = pipeline.process_image(img_path)
            pipeline.export_json(res, os.path.join(output_dir, f"{base_name}_result.json"))

            sev_val = res.severity.severity_pct if res.severity else None
            conf_val = res.severity.confidence_score if res.severity else None

            rows.append({
                "Filename": os.path.basename(img_path),
                "Status": res.status.value,
                "Quality": res.quality.status.value,
                "Mean_Lightness": res.quality.mean_lightness,
                "Laplacian_Var": res.quality.laplacian_variance,
                "Severity_Pct": sev_val,
                "Confidence_Score": conf_val,
                "Latency_ms": res.performance.latency_mean_ms
            })

    if export_csv and rows:
        df = pd.DataFrame(rows)
        csv_path = os.path.join(output_dir, "batch_summary.csv")
        df.to_csv(csv_path, index=False)
        console.print(f"[bold green]✓ Batch CSV summary exported to:[/bold green] {csv_path}")


@cli.command("camera")
@click.option("--camera-index", "-c", default=0, help="Camera device index (0 for primary webcam)")
@click.option("--fps-limit", "-f", default=15.0, help="Target FPS frame rate limit")
@click.option("--config", default="configs/default_config.yaml", help="Configuration file")
def camera_stream_cmd(camera_index: int, fps_limit: float, config_path: str):
    """Run live camera stream processing with real-time severity HUD overlay."""
    console.print(Panel.fit(f"[bold green]FieldSight-Lite: Live Camera Feed (Device #{camera_index})[/bold green]"))
    console.print("[dim]Press 'q' in the OpenCV window to exit stream.[/dim]")

    pipeline = FieldSightPipeline(config_path)
    cap = CameraCapture(source=camera_index, target_fps=fps_limit)

    if not cap.open():
        console.print(f"[red]Error: Could not open camera device index {camera_index}[/red]")
        return

    try:
        while True:
            ret, frame = cap.read_frame()
            if not ret or frame is None:
                continue

            res = pipeline.process_image(frame)
            display_frame = res.overlay_image if res.overlay_image is not None else frame.copy()

            # Render HUD overlay text
            if res.severity is not None:
                hud_text = f"Severity: {res.severity.severity_pct:.1f}% | Conf: {res.severity.confidence_score:.0f}% | {res.performance.fps:.1f} FPS"
                color = (0, 0, 255) if res.severity.severity_pct > 5.0 else (0, 255, 0)
            else:
                hud_text = f"Status: {res.status.value} | {res.quality.status.value}"
                color = (0, 165, 255)

            cv2.putText(display_frame, hud_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.imshow("FieldSight-Lite Live Edge Monitor", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        console.print("[green]Camera stream released successfully.[/green]")


@cli.command("benchmark")
@click.option("--data-dir", default=None, help="Optional image dataset directory")
@click.option("--config", default="configs/default_config.yaml", help="Configuration file")
@click.option("--export-latex", default="data/benchmark_results/table.tex", help="LaTeX output path")
@click.option("--export-csv", default="data/benchmark_results/summary.csv", help="CSV output path")
def benchmark_cmd(data_dir: Optional[str], config: str, export_latex: str, export_csv: str):
    """Run full comparative benchmark vs Baselines and export LaTeX tables."""
    console.print(Panel.fit("[bold green]FieldSight-Lite: Comparative Benchmark Suite[/bold green]"))

    dataset = create_synthetic_benchmark_dataset(num_plants=8, leaves_per_plant=4, seed=42)
    _, _, test_set = split_by_plant_id(dataset, train_ratio=0.5, val_ratio=0.2, seed=42)
    console.print(f"Evaluating [bold cyan]{len(test_set)}[/bold cyan] test leaves across 5 lighting regimes...")

    pipeline = FieldSightPipeline(config)
    df = run_benchmark_suite(test_set, pipeline)

    console.print("\n=== BENCHMARK COMPARATIVE RESULTS ===")
    console.print(df.to_string(index=False))

    os.makedirs(os.path.dirname(os.path.abspath(export_csv)), exist_ok=True)
    df.to_csv(export_csv, index=False)
    console.print(f"\n[green]✓ Saved CSV summary to:[/green] {export_csv}")

    export_latex_table(df, export_latex)
    console.print(f"[green]✓ Saved LaTeX publication table to:[/green] {export_latex}")


@cli.command("progression")
@click.option("--plant-id", required=True, help="Plant identifier")
@click.option("--leaf-id", required=True, help="Leaf identifier")
@click.option("--db-path", default="data/temporal_db/field_history.db", help="Path to SQLite DB")
def progression_cmd(plant_id: str, leaf_id: str, db_path: str):
    """Query temporal progression history and growth velocity for a leaf."""
    console.print(Panel.fit(f"[bold green]Temporal Progression for Plant {plant_id} / Leaf {leaf_id}[/bold green]"))

    ledger = ProgressionLedger(db_path=db_path)
    report = ledger.analyze_progression(plant_id, leaf_id)

    if not report.observation_history:
        console.print(f"[yellow]No observation history found for Plant: {plant_id}, Leaf: {leaf_id}[/yellow]")
        return

    table = Table(title=f"Observation History (Total: {len(report.observation_history)})")
    table.add_column("Timestamp (UTC)", style="cyan")
    table.add_column("Severity (%)", style="magenta")
    table.add_column("Confidence (%)", style="green")

    for obs in report.observation_history:
        table.add_row(obs.timestamp_utc, f"{obs.severity_pct:.2f}%", f"{obs.confidence_score:.1f}%")

    console.print(table)

    rate_str = f"{report.progression_rate_daily:.2f} %/day" if report.progression_rate_daily is not None else "N/A"
    summary_table = Table(title="Progression Diagnostics")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="bold yellow")
    summary_table.add_row("Delta Time (Days)", f"{report.delta_days:.2f}")
    summary_table.add_row("Delta Severity", f"{report.delta_severity:.2f}%")
    summary_table.add_row("Progression Rate", rate_str)
    summary_table.add_row("Progression State", str(report.state.value))
    summary_table.add_row("Risk Alert Level", str(report.risk_alert_level.value))

    console.print(summary_table)


if __name__ == "__main__":
    cli()
