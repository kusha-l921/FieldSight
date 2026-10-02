"""
Automated benchmark runner and LaTeX publication table exporter for FieldSight-Lite.
Evaluates Proposed method vs Baselines across clean and perturbed lighting regimes.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

from experiments.baselines import (
    GlobalOtsuBaseline,
    HSVThresholdBaseline,
    KMeansColorBaseline,
    RGBThresholdBaseline,
)
from experiments.dataset_loader import (
    BenchmarkSample,
    create_synthetic_benchmark_dataset,
    split_by_plant_id,
)
from experiments.evaluation import evaluate_predictions
from experiments.synthetic_perturb import generate_all_perturbations
from src.pipeline import FieldSightPipeline
from src.schemas import EvaluationMetrics


def run_benchmark_suite(
    test_samples: List[BenchmarkSample],
    pipeline: FieldSightPipeline
) -> pd.DataFrame:
    """
    Executes benchmark comparison across Proposed FieldSight-Lite and 4 Baselines.
    Evaluates both Clean condition and Perturbed stress conditions.
    """
    baselines = {
        "Proposed (FieldSight-Lite)": None,  # Handled via pipeline
        "Baseline 1: RGB Naive": RGBThresholdBaseline(),
        "Baseline 2: HSV Color": HSVThresholdBaseline(),
        "Baseline 3: Global Otsu": GlobalOtsuBaseline(),
        "Baseline 4: K-Means (K=3)": KMeansColorBaseline(),
    }

    records: List[Dict[str, Any]] = []

    for model_name, model_obj in baselines.items():
        # 1. Clean Evaluation
        clean_leaf_preds, clean_les_preds, clean_sev_preds = [], [], []
        gt_leafs, gt_lesions, gt_sevs = [], [], []

        # 2. Perturbed Evaluation
        pert_leaf_preds, pert_les_preds, pert_sev_preds = [], [], []
        pert_gt_leafs, pert_gt_lesions, pert_gt_sevs = [], [], []

        for sample in test_samples:
            gt_leaf = sample.gt_leaf_mask
            gt_les = sample.gt_lesion_mask
            gt_s = sample.gt_severity_pct

            # Predict Clean
            if model_name == "Proposed (FieldSight-Lite)":
                res = pipeline.process_image(sample.image_bgr)
                p_leaf = res.segmentation.leaf_mask if res.segmentation else np.zeros_like(gt_leaf)
                p_les = res.segmentation.lesion_mask if res.segmentation else np.zeros_like(gt_les)
                p_s = res.severity.severity_pct if res.severity else 0.0
            else:
                p_leaf, p_les, p_s = model_obj.predict(sample.image_bgr)

            clean_leaf_preds.append(p_leaf)
            clean_les_preds.append(p_les)
            clean_sev_preds.append(p_s)
            gt_leafs.append(gt_leaf)
            gt_lesions.append(gt_les)
            gt_sevs.append(gt_s)

            # Predict Perturbed
            perts = generate_all_perturbations(sample.image_bgr)
            for _, pert_img in perts.items():
                if model_name == "Proposed (FieldSight-Lite)":
                    p_res = pipeline.process_image(pert_img)
                    pt_leaf = p_res.segmentation.leaf_mask if p_res.segmentation else np.zeros_like(gt_leaf)
                    pt_les = p_res.segmentation.lesion_mask if p_res.segmentation else np.zeros_like(gt_les)
                    pt_s = p_res.severity.severity_pct if p_res.severity else 0.0
                else:
                    pt_leaf, pt_les, pt_s = model_obj.predict(pert_img)

                pert_leaf_preds.append(pt_leaf)
                pert_les_preds.append(pt_les)
                pert_sev_preds.append(pt_s)
                pert_gt_leafs.append(gt_leaf)
                pert_gt_lesions.append(gt_les)
                pert_gt_sevs.append(gt_s)

        # Compute Metrics
        m_clean = evaluate_predictions(clean_leaf_preds, clean_les_preds, clean_sev_preds, gt_leafs, gt_lesions, gt_sevs)
        m_pert = evaluate_predictions(pert_leaf_preds, pert_les_preds, pert_sev_preds, pert_gt_leafs, pert_gt_lesions, pert_gt_sevs)

        records.append({
            "Method": model_name,
            "Clean Leaf IoU (%)": round(m_clean.leaf_iou * 100.0, 2),
            "Clean Lesion IoU (%)": round(m_clean.lesion_iou * 100.0, 2),
            "Clean Sev MAE (%)": round(m_clean.severity_mae, 2),
            "Perturbed Leaf IoU (%)": round(m_pert.leaf_iou * 100.0, 2),
            "Perturbed Lesion IoU (%)": round(m_pert.lesion_iou * 100.0, 2),
            "Perturbed Sev MAE (%)": round(m_pert.severity_mae, 2),
            "Robustness Drop (ΔMAE)": round(m_pert.severity_mae - m_clean.severity_mae, 2)
        })

    return pd.DataFrame(records)


def export_latex_table(df: pd.DataFrame, output_path: str) -> str:
    """
    Generates publication-quality LaTeX table from benchmark dataframe.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    latex_code = [
        "\\begin{table*}[htbp]",
        "\\centering",
        "\\caption{Quantitative Benchmark: FieldSight-Lite vs. Baselines under Clean and Severe Perturbed Lighting Regimes.}",
        "\\label{tab:fieldsight_benchmarks}",
        "\\small",
        "\\begin{tabular}{lcccccc|c}",
        "\\hline",
        "\\textbf{Method} & \\multicolumn{3}{c}{\\textbf{Clean Lighting}} & \\multicolumn{3}{c}{\\textbf{Perturbed Lighting (Stress Test)}} & \\textbf{Drift} \\\\",
        " & \\textbf{Leaf IoU} & \\textbf{Lesion IoU} & \\textbf{Sev MAE} & \\textbf{Leaf IoU} & \\textbf{Lesion IoU} & \\textbf{Sev MAE} & \\textbf{$\\Delta$MAE} \\\\",
        "\\hline"
    ]

    for _, row in df.iterrows():
        is_proposed = "Proposed" in str(row["Method"])
        prefix = "\\textbf{" if is_proposed else ""
        suffix = "}" if is_proposed else ""
        
        line = (
            f"{prefix}{row['Method']}{suffix} & "
            f"{prefix}{row['Clean Leaf IoU (%)']:.1f}\\%{suffix} & "
            f"{prefix}{row['Clean Lesion IoU (%)']:.1f}\\%{suffix} & "
            f"{prefix}{row['Clean Sev MAE (%)']:.2f}\\%{suffix} & "
            f"{prefix}{row['Perturbed Leaf IoU (%)']:.1f}\\%{suffix} & "
            f"{prefix}{row['Perturbed Lesion IoU (%)']:.1f}\\%{suffix} & "
            f"{prefix}{row['Perturbed Sev MAE (%)']:.2f}\\%{suffix} & "
            f"{prefix}{row['Robustness Drop (ΔMAE)']:.2f}\\%{suffix} \\\\"
        )
        latex_code.append(line)

    latex_code.extend([
        "\\hline",
        "\\end{tabular}",
        "\\end{table*}"
    ])

    full_latex = "\n".join(latex_code)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_latex)
    return full_latex


def main():
    parser = argparse.ArgumentParser(description="Run FieldSight-Lite Benchmarks")
    parser.add_argument("--config", default="configs/default_config.yaml", help="Path to config file")
    parser.add_argument("--num-plants", type=int, default=8, help="Synthetic test plants")
    parser.add_argument("--export-latex", default="data/benchmark_results/table.tex", help="LaTeX output path")
    parser.add_argument("--export-csv", default="data/benchmark_results/summary.csv", help="CSV output path")
    args = parser.parse_args()

    print("Generating synthetic benchmark cohort with strict Plant-ID separation...")
    dataset = create_synthetic_benchmark_dataset(num_plants=args.num_plants, leaves_per_plant=4, seed=42)
    _, _, test_set = split_by_plant_id(dataset, train_ratio=0.5, val_ratio=0.2, seed=42)
    print(f"Evaluating on {len(test_set)} unseen test leaves...")

    pipeline = FieldSightPipeline(args.config)
    df = run_benchmark_suite(test_set, pipeline)

    print("\n=== BENCHMARK RESULTS SUMMARY ===")
    print(df.to_string(index=False))

    os.makedirs(os.path.dirname(os.path.abspath(args.export_csv)), exist_ok=True)
    df.to_csv(args.export_csv, index=False)
    print(f"\nSaved CSV summary to: {args.export_csv}")

    latex_str = export_latex_table(df, args.export_latex)
    print(f"Saved LaTeX publication table to: {args.export_latex}")


if __name__ == "__main__":
    main()
