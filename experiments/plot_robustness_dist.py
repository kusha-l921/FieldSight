"""
Publication-Grade Empirical Robustness Distribution Plotting.
Generates:
- Figure 4A (results/plots/fig4_masd_lrs_distribution.png): Side-by-side violin/box plots
  displaying empirical distributions of Lighting Robustness Score (LRS, %) and
  Mean Absolute Severity Drift (MASD, %) across real tomato leaves.
- Figure 4B (results/plots/fig4_perturbation_drift_breakdown.png): Comparative box plot
  breaking down severity drift delta across each of the 5 environmental lighting perturbations.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PERTURBATION_DISPLAY_NAMES = {
    "drift_shadow_ramp": "Shadow Ramp\n(min=0.35x)",
    "drift_specular_glare": "Specular Glare\n(hotspot=160)",
    "drift_overexposure": "Overexposure\n(+55 bias)",
    "drift_underexposure": "Underexposure\n(0.50x scale)",
    "drift_gamma_shift": "Gamma Shift\n(γ=1.80)",
}


def setup_publication_style():
    """Configures modern, publication-ready typography, grids, and aesthetic theme."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 11,
        "figure.titlesize": 14,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.0,
        "grid.color": "#e0e0e0",
        "grid.linestyle": "--",
        "grid.alpha": 0.6,
    })


def plot_fig4a_masd_lrs_distribution(
    df: pd.DataFrame,
    output_path: str = "results/plots/fig4_masd_lrs_distribution.png"
) -> None:
    """
    Plots Figure 4A: Side-by-side violin and box plots showing the empirical distribution
    of LRS (%) and MASD (%) across evaluated real leaves.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    setup_publication_style()

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.5))
    n = len(df)

    # -------------------------------------------------------------
    # Subplot 1: Lighting Robustness Score (LRS)
    # -------------------------------------------------------------
    ax1 = axes[0]
    lrs = df["lighting_robustness_score"]
    lrs_mean, lrs_std = lrs.mean(), lrs.std()
    lrs_med, lrs_iqr = lrs.median(), lrs.quantile(0.75) - lrs.quantile(0.25)

    # Violin plot
    v1 = ax1.violinplot(
        lrs,
        positions=[1],
        widths=0.65,
        showmeans=False,
        showmedians=False,
        showextrema=False
    )
    for pc in v1["bodies"]:
        pc.set_facecolor("#2E7D32")  # Deep Emerald Green
        pc.set_edgecolor("#1B5E20")
        pc.set_alpha(0.35)
        pc.set_linewidth(1.5)

    # Inner Boxplot
    box1 = ax1.boxplot(
        lrs,
        positions=[1],
        widths=0.22,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        showfliers=False,
        boxprops=dict(facecolor="#4CAF50", edgecolor="#1B5E20", linewidth=1.4, alpha=0.85),
        medianprops=dict(color="#FFFFFF", linewidth=2.5),
        meanprops=dict(color="#FFD600", linewidth=2.0, linestyle="--"),
        whiskerprops=dict(color="#1B5E20", linewidth=1.3),
        capprops=dict(color="#1B5E20", linewidth=1.3)
    )

    # Jittered strip scatter points
    np.random.seed(42)
    jitter1 = np.random.normal(0, 0.04, size=len(lrs))
    ax1.scatter(
        1.0 + jitter1,
        lrs,
        color="#1B5E20",
        alpha=0.55,
        s=28,
        edgecolors="white",
        linewidth=0.5,
        zorder=5,
        label="Leaves"
    )

    ax1.set_xlim(0.4, 1.6)
    ax1.set_xticks([1])
    ax1.set_xticklabels([f"Real Foliage Cohort\n(N={n})"], fontweight="semibold")
    ax1.set_ylabel("Lighting Robustness Score (LRS, %)", fontweight="semibold")
    ax1.set_title("Distribution of LRS", fontweight="bold", pad=10)

    # Statistical summary callout
    callout_text_1 = (
        f"Mean: {lrs_mean:.2f}% ± {lrs_std:.2f}%\n"
        f"Median: {lrs_med:.2f}%\n"
        f"IQR: {lrs_iqr:.2f}%\n"
        f"Min / Max: {lrs.min():.1f}% / {lrs.max():.1f}%"
    )
    ax1.text(
        0.05, 0.06, callout_text_1,
        transform=ax1.transAxes,
        fontsize=9.5,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#E8F5E9", edgecolor="#81C784", alpha=0.9)
    )

    # -------------------------------------------------------------
    # Subplot 2: Mean Absolute Severity Drift (MASD)
    # -------------------------------------------------------------
    ax2 = axes[1]
    masd = df["mean_severity_drift"]
    masd_mean, masd_std = masd.mean(), masd.std()
    masd_med, masd_iqr = masd.median(), masd.quantile(0.75) - masd.quantile(0.25)

    # Violin plot
    v2 = ax2.violinplot(
        masd,
        positions=[1],
        widths=0.65,
        showmeans=False,
        showmedians=False,
        showextrema=False
    )
    for pc in v2["bodies"]:
        pc.set_facecolor("#C62828")  # Vivid Crimson Red
        pc.set_edgecolor("#8E0000")
        pc.set_alpha(0.35)
        pc.set_linewidth(1.5)

    # Inner Boxplot
    box2 = ax2.boxplot(
        masd,
        positions=[1],
        widths=0.22,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        showfliers=False,
        boxprops=dict(facecolor="#EF5350", edgecolor="#8E0000", linewidth=1.4, alpha=0.85),
        medianprops=dict(color="#FFFFFF", linewidth=2.5),
        meanprops=dict(color="#FFD600", linewidth=2.0, linestyle="--"),
        whiskerprops=dict(color="#8E0000", linewidth=1.3),
        capprops=dict(color="#8E0000", linewidth=1.3)
    )

    # Jittered strip scatter points
    jitter2 = np.random.normal(0, 0.04, size=len(masd))
    ax2.scatter(
        1.0 + jitter2,
        masd,
        color="#8E0000",
        alpha=0.55,
        s=28,
        edgecolors="white",
        linewidth=0.5,
        zorder=5
    )

    ax2.set_xlim(0.4, 1.6)
    ax2.set_xticks([1])
    ax2.set_xticklabels([f"Real Foliage Cohort\n(N={n})"], fontweight="semibold")
    ax2.set_ylabel("Mean Absolute Severity Drift (MASD, %)", fontweight="semibold")
    ax2.set_title("Distribution of MASD", fontweight="bold", pad=10)

    # Statistical summary callout
    callout_text_2 = (
        f"Mean: {masd_mean:.2f}% ± {masd_std:.2f}%\n"
        f"Median: {masd_med:.2f}%\n"
        f"IQR: {masd_iqr:.2f}%\n"
        f"Min / Max: {masd.min():.1f}% / {masd.max():.1f}%"
    )
    ax2.text(
        0.95, 0.94, callout_text_2,
        transform=ax2.transAxes,
        fontsize=9.5,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFEBEE", edgecolor="#E57373", alpha=0.9)
    )

    # Global title and legend
    fig.suptitle(
        "Figure 4A: Empirical Robustness Distributions across Real Tomato Leaves (N=100)",
        fontsize=13.5,
        fontweight="bold",
        y=0.98
    )

    # Custom legend for markers
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#FFFFFF", lw=2.5, label="Median (white line)"),
        Line2D([0], [0], color="#FFD600", lw=2.0, linestyle="--", label="Mean (yellow dash)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#555555", markersize=6, label="Individual Leaf")
    ]
    fig.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=3, frameon=True, fontsize=9.5)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 4A to: {output_path}")


def plot_fig4b_perturbation_drift_breakdown(
    df: pd.DataFrame,
    output_path: str = "results/plots/fig4_perturbation_drift_breakdown.png"
) -> None:
    """
    Plots Figure 4B: Box plot comparing severity drift delta across each of the 5
    individual lighting perturbations to reveal which environmental condition causes
    the largest variance.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    setup_publication_style()

    # Filter columns that are present
    valid_cols = [c for c in PERTURBATION_DISPLAY_NAMES.keys() if c in df.columns]
    if not valid_cols:
        print("Warning: No perturbation drift columns found in dataframe.")
        return

    # Create long-format dataframe for seaborn
    data_list = []
    for col in valid_cols:
        display_name = PERTURBATION_DISPLAY_NAMES[col]
        for val in df[col]:
            data_list.append({
                "Perturbation": display_name,
                "Drift": val
            })
    df_long = pd.DataFrame(data_list)

    # Determine condition with maximum mean drift
    means_by_pert = df_long.groupby("Perturbation", sort=False)["Drift"].mean()
    max_pert = means_by_pert.idxmax()
    max_mean_val = means_by_pert.max()

    fig, ax = plt.subplots(figsize=(11.5, 6.0))

    # Distinct harmonious palette for 5 conditions
    palette = ["#3949AB", "#FB8C00", "#E53935", "#039BE5", "#8E24AA"]

    # Boxplot
    sns.boxplot(
        data=df_long,
        x="Perturbation",
        y="Drift",
        hue="Perturbation",
        legend=False,
        palette=palette,
        ax=ax,
        width=0.48,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "yellow",
            "markeredgecolor": "black",
            "markersize": 7,
            "label": "Mean Drift"
        },
        medianprops={"color": "white", "linewidth": 2.2},
        boxprops={"alpha": 0.85, "linewidth": 1.2},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
        fliersize=0  # Hide fliers in boxplot as stripplot overlays all points
    )

    # Stripplot overlay for individual leaf observations
    sns.stripplot(
        data=df_long,
        x="Perturbation",
        y="Drift",
        color="#212121",
        alpha=0.35,
        size=4.5,
        jitter=0.20,
        ax=ax
    )

    ax.set_ylabel("Absolute Severity Drift ΔS = |S_pert - S_0| (%)", fontweight="semibold")
    ax.set_xlabel("Physical Environmental Lighting Stress Regime", fontweight="semibold", labelpad=10)
    ax.set_title(
        "Figure 4B: Environmental Lighting Perturbation Stress Breakdown across Real Leaves (N=100)",
        fontweight="bold",
        pad=14
    )

    # Annotate the perturbation causing largest variance/drift
    pert_names_list = [PERTURBATION_DISPLAY_NAMES[c] for c in valid_cols]
    if max_pert in pert_names_list:
        idx = pert_names_list.index(max_pert)
        ax.annotate(
            f"Largest Drift\n(Mean: {max_mean_val:.2f}%)",
            xy=(idx, max_mean_val),
            xytext=(idx + 0.35, max_mean_val + 1.2),
            arrowprops=dict(facecolor="#D32F2F", shrink=0.08, width=1.5, headwidth=7),
            fontsize=9.5,
            fontweight="bold",
            color="#B71C1C",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFEBEE", edgecolor="#E57373", alpha=0.9)
        )

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="white", lw=2.2, label="Median (white line)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="yellow", markeredgecolor="black", markersize=8, label="Mean Drift"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#424242", markersize=6, alpha=0.5, label="Individual Leaf ΔS")
    ]
    ax.legend(handles=legend_elements, loc="upper right", frameon=True, fontsize=9.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 4B to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Empirical Robustness Distribution Figures (Fig 4A & 4B)")
    parser.add_argument(
        "--input-csv",
        default="data/benchmark_results/real_stress_test_metrics.csv",
        help="Path to compiled real stress test CSV"
    )
    parser.add_argument(
        "--output-dir",
        default="results/plots",
        help="Directory where generated figures are saved"
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_csv):
        print(f"Error: Metrics CSV not found at: {args.input_csv}")
        print("Please run experiments/batch_stress_test.py first to generate empirical results.")
        sys.exit(1)

    df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df)} records from {args.input_csv}")

    fig4a_path = os.path.join(args.output_dir, "fig4_masd_lrs_distribution.png")
    fig4b_path = os.path.join(args.output_dir, "fig4_perturbation_drift_breakdown.png")

    plot_fig4a_masd_lrs_distribution(df, fig4a_path)
    plot_fig4b_perturbation_drift_breakdown(df, fig4b_path)
    print("All empirical distribution figures generated successfully.")


if __name__ == "__main__":
    main()
