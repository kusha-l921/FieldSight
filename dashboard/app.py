"""
FieldSight-Lite: Interactive Research & Edge Deployment Dashboard.
A 4-Tab Streamlit application for leaf disease severity quantification,
active lighting stress-testing, temporal progression velocity tracking, and benchmark evaluation.
"""

from __future__ import annotations

import io
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add workspace to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.dataset_loader import (
    generate_synthetic_leaf,
    create_synthetic_benchmark_dataset,
    split_by_plant_id,
)
from experiments.run_benchmarks import export_latex_table, run_benchmark_suite
from experiments.synthetic_perturb import generate_all_perturbations
from src.pipeline import FieldSightPipeline
from src.progression import ProgressionLedger
from src.schemas import ProgressionState, RiskAlertLevel

# Page configuration
st.set_page_config(
    page_title="FieldSight-Lite | Edge Plant Disease Platform",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.08) 0%, rgba(5, 150, 105, 0.03) 100%);
        border: 1px solid rgba(16, 185, 129, 0.2);
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.03);
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 500;
        color: #4b5563;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #065f46;
        margin-top: 4px;
    }
    .badge {
        display: inline-block;
        padding: 4px 10px;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 20px;
    }
    .badge-stable { background-color: #d1fae5; color: #065f46; }
    .badge-emerging { background-color: #fef3c7; color: #92400e; }
    .badge-moderate { background-color: #ffedd5; color: #9a3412; }
    .badge-rapid { background-color: #fee2e2; color: #991b1b; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_pipeline(config_path: str = "configs/default_config.yaml") -> FieldSightPipeline:
    return FieldSightPipeline(config_path)


@st.cache_resource
def get_ledger(db_path: str = "data/temporal_db/field_history.db") -> ProgressionLedger:
    return ProgressionLedger(db_path=db_path)


# Sidebar Configuration
st.sidebar.image("https://img.icons8.com/isometric/512/leaf.png", width=64)
st.sidebar.title("FieldSight-Lite")
st.sidebar.markdown("**Illumination-Robust Edge Monitoring**")
st.sidebar.markdown("---")

config_path = st.sidebar.text_input("Config Path", "configs/default_config.yaml")
db_path = st.sidebar.text_input("SQLite DB Path", "data/temporal_db/field_history.db")

pipeline = get_pipeline(config_path)
ledger = get_ledger(db_path)

# Main Title Header
st.title("🌿 FieldSight-Lite Platform")
st.caption("Deterministic, Training-Free Plant Disease Severity & Progression Quantification on Edge Devices")

# Tab Navigation
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 1. Foliar Inspection",
    "⚡ 2. Active Lighting Stress Test",
    "📈 3. Temporal Progression Ledger",
    "📊 4. Benchmarks & LaTeX Export"
])

# -------------------------------------------------------------
# TAB 1: Foliar Inspection
# -------------------------------------------------------------
with tab1:
    st.subheader("Foliar Leaf Disease Analysis & Optical Quality Gate")

    col_input, col_meta = st.columns([2, 1])

    with col_input:
        input_source = st.radio(
            "Input Image Source",
            ["Preset Sample Leaves", "Upload Custom Leaf Image"],
            horizontal=True
        )

        image_bgr: Optional[np.ndarray] = None

        if input_source == "Preset Sample Leaves":
            preset_choice = st.selectbox(
                "Choose Foliar Pathological Condition",
                [
                    "Healthy Leaf (0.0% Severity)",
                    "Early Foliar Scab (5.5% Severity)",
                    "Moderate Rust / Blight (14.2% Severity)",
                    "Severe Necrotic Spot (28.6% Severity)",
                    "Simulated Specular Glare (Quality Gate Test)",
                    "Simulated Motion Blur (Quality Gate Test)"
                ]
            )

            if "Healthy" in preset_choice:
                img, _, _, _ = generate_synthetic_leaf(severity_target_pct=0.0, seed=101)
                image_bgr = img
            elif "Early" in preset_choice:
                img, _, _, _ = generate_synthetic_leaf(severity_target_pct=5.5, seed=202)
                image_bgr = img
            elif "Moderate" in preset_choice:
                img, _, _, _ = generate_synthetic_leaf(severity_target_pct=14.2, seed=303)
                image_bgr = img
            elif "Severe" in preset_choice:
                img, _, _, _ = generate_synthetic_leaf(severity_target_pct=28.6, seed=404)
                image_bgr = img
            elif "Glare" in preset_choice:
                img, _, _, _ = generate_synthetic_leaf(severity_target_pct=10.0, seed=505)
                # Apply harsh glare to fail preflight
                h, w = img.shape[:2]
                cv2.circle(img, (w // 2, h // 2), int(w * 0.35), (255, 255, 255), -1)
                image_bgr = img
            elif "Blur" in preset_choice:
                img, _, _, _ = generate_synthetic_leaf(severity_target_pct=10.0, seed=606)
                image_bgr = cv2.GaussianBlur(img, (25, 25), 15.0)
        else:
            uploaded = st.file_uploader("Upload Leaf Image (JPEG/PNG/WebP)", type=["jpg", "jpeg", "png", "webp"])
            if uploaded is not None:
                file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
                image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    with col_meta:
        st.markdown("**Observation Metadata**")
        p_id = st.text_input("Plant Identifier", value="PLANT_001")
        l_id = st.text_input("Leaf Identifier", value="LEAF_01")
        record_to_db = st.checkbox("Log directly to Progression DB", value=True)

    if image_bgr is not None:
        # Run Pipeline
        res = pipeline.process_image(
            image_input=image_bgr,
            plant_id=p_id if record_to_db else None,
            leaf_id=l_id if record_to_db else None
        )

        st.markdown("---")
        # Visual Multi-panel Artifacts
        st.markdown("### 🖼️ Multi-Channel Visual Decomposition")

        v_col1, v_col2, v_col3, v_col4, v_col5 = st.columns(5)

        # 1. Raw BGR
        v_col1.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), caption="1. Raw Input Image", use_container_width=True)

        # 2. L* Lightness View
        if res.quality.is_valid_for_processing:
            lab_norm, l_norm, _ = pipeline.normalizer.normalize(image_bgr)
            v_col2.image(l_norm, caption="2. CLAHE Lightness (L*)", use_container_width=True, clamp=True)
        else:
            v_col2.info("Quality gate failed")

        # 3. Leaf Mask
        if res.segmentation is not None:
            v_col3.image(res.segmentation.leaf_mask, caption="3. Leaf Mask (M_leaf)", use_container_width=True)
            v_col4.image(res.segmentation.lesion_mask, caption="4. Lesion Mask (M_lesion)", use_container_width=True)
            v_col5.image(cv2.cvtColor(res.overlay_image, cv2.COLOR_BGR2RGB), caption="5. Heatmap Overlay", use_container_width=True)
        else:
            v_col3.warning("No Leaf Mask")
            v_col4.warning("No Lesion Mask")
            v_col5.warning("No Overlay")

        st.markdown("---")
        # KPI Metric Cards
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

        with kpi1:
            if res.severity is not None:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Disease Severity</div>
                    <div class="metric-value" style="color: #dc2626;">{res.severity.severity_pct:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Disease Severity</div>
                    <div class="metric-value" style="color: #6b7280;">N/A</div>
                </div>
                """, unsafe_allow_html=True)

        with kpi2:
            if res.severity is not None:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Unaffected Foliage</div>
                    <div class="metric-value" style="color: #059669;">{res.severity.unaffected_pct:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Unaffected Foliage</div>
                    <div class="metric-value" style="color: #6b7280;">N/A</div>
                </div>
                """, unsafe_allow_html=True)

        with kpi3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Quality Gate</div>
                <div class="metric-value" style="font-size: 1.25rem;">{res.quality.status.value}</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi4:
            conf_str = f"{res.severity.confidence_score:.1f}%" if res.severity is not None else "0.0%"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Confidence Score</div>
                <div class="metric-value" style="color: #2563eb;">{conf_str}</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Edge Latency</div>
                <div class="metric-value" style="font-size: 1.35rem;">{res.performance.latency_mean_ms:.1f} ms <span style="font-size: 0.9rem; color: #4b5563;">({res.performance.fps:.1f} FPS)</span></div>
            </div>
            """, unsafe_allow_html=True)

        # Baseline Inlier Statistics
        if res.segmentation is not None:
            with st.expander("🔬 View Robust Inlier Chromaticity Parameters (Pass 2 Statistics)"):
                s_c1, s_c2, s_c3, s_c4 = st.columns(4)
                s_c1.metric("Healthy Median a*", f"{res.segmentation.healthy_mu_a:.2f}")
                s_c2.metric("Healthy MAD a*", f"{res.segmentation.healthy_mad_a:.2f}")
                s_c3.metric("Healthy Median b*", f"{res.segmentation.healthy_mu_b:.2f}")
                s_c4.metric("Healthy MAD b*", f"{res.segmentation.healthy_mad_b:.2f}")

        # JSON Export Download
        json_str = pipeline.export_result_dict(res)
        st.download_button(
            label="📥 Download Structured JSON Result (result.json)",
            data=pd.Series(json_str).to_json(indent=2),
            file_name=f"{p_id}_{l_id}_result.json",
            mime="application/json"
        )


# -------------------------------------------------------------
# TAB 2: Active Lighting Stress Test
# -------------------------------------------------------------
with tab2:
    st.subheader("⚡ Active Optical Lighting Stress-Testing Engine")
    st.caption("Applies 5 deterministic physical transformations to quantify Lighting Robustness Score (LRS) and Severity Drift (MASD).")

    if image_bgr is None:
        st.info("Please select or upload an image in Tab 1 first.")
    else:
        if st.button("🚀 Run Active 5-Perturbation Stress Suite", type="primary"):
            with st.spinner("Subjecting leaf to physical illumination stress..."):
                stress_res = pipeline.process_image(
                    image_input=image_bgr,
                    run_stress_test=True
                )

            if stress_res.robustness is not None:
                r_rep = stress_res.robustness

                st.markdown("### 🏆 Robustness Scorecard")
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("Lighting Robustness Score (LRS)", f"{r_rep.lighting_robustness_score:.1f}%")
                r2.metric("Mean Absolute Severity Drift (MASD)", f"{r_rep.mean_severity_drift:.2f}%")
                r3.metric("Severity Std Dev (σ)", f"{r_rep.severity_std_dev:.2f}%")
                r4.metric("Field Robustness Index (FRI)", f"{r_rep.field_robustness_index:.1f}%")

                st.markdown("---")
                st.markdown("### 🔬 Perturbation Breakdown & Mask Stability")

                p_cols = st.columns(len(r_rep.perturbation_breakdown))
                perts = generate_all_perturbations(image_bgr)

                drift_data = [{"Condition": "Baseline (S0)", "Severity (%)": r_rep.original_severity}]

                for idx, p in enumerate(r_rep.perturbation_breakdown):
                    with p_cols[idx]:
                        pert_img = perts.get(p.perturbation_name)
                        if pert_img is not None:
                            st.image(cv2.cvtColor(pert_img, cv2.COLOR_BGR2RGB), caption=p.perturbation_name, use_container_width=True)
                        st.markdown(f"**Severity:** `{p.perturbed_severity_pct:.2f}%`")
                        st.markdown(f"**Drift:** `Δ {p.severity_drift_delta:.2f}%`")
                        st.markdown(f"**Lesion IoU:** `{p.lesion_mask_stability_iou * 100.0:.1f}%`")

                    drift_data.append({
                        "Condition": p.perturbation_name,
                        "Severity (%)": p.perturbed_severity_pct
                    })

                # Plotly Drift Chart
                df_drift = pd.DataFrame(drift_data)
                fig_drift = px.bar(
                    df_drift,
                    x="Condition",
                    y="Severity (%)",
                    color="Severity (%)",
                    color_continuous_scale="Reds",
                    title="Severity Percentage Stability across Environmental Lighting Variations"
                )
                fig_drift.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_drift, use_container_width=True)


# -------------------------------------------------------------
# TAB 3: Temporal Progression Ledger
# -------------------------------------------------------------
with tab3:
    st.subheader("📈 Longitudinal SQLite Progression Ledger & Growth Velocity")
    st.caption("Tracks multi-day disease progression (ΔS/day) and raises epidemiological alerts.")

    entities = ledger.list_monitored_entities()

    p_col1, p_col2 = st.columns([1, 2])

    with p_col1:
        st.markdown("#### Tracked Plant / Leaf Selector")
        if entities:
            selected_ent = st.selectbox(
                "Select Monitored Leaf",
                options=entities,
                format_func=lambda x: f"Plant {x[0]} - Leaf {x[1]} ({x[2]} obs)"
            )
            curr_plant_id, curr_leaf_id = selected_ent[0], selected_ent[1]
        else:
            curr_plant_id = st.text_input("Monitored Plant ID", value="PLANT_FIELD_01")
            curr_leaf_id = st.text_input("Monitored Leaf ID", value="LEAF_A")

        if st.button("🌱 Populate 14-Day Mock Field Progression"):
            base_time = datetime.now(timezone.utc) - timedelta(days=14)
            # Simulate sigmoidal disease progression
            for day in range(15):
                obs_time = (base_time + timedelta(days=day)).strftime("%Y-%m-%dT%H:%M:%SZ")
                # Sigmoid curve: 0.5% -> 22.0%
                sev = float(22.0 / (1.0 + np.exp(-0.4 * (day - 7))))
                conf = float(92.0 + np.random.uniform(-3, 3))
                ledger.record_observation(
                    plant_id=curr_plant_id,
                    leaf_id=curr_leaf_id,
                    severity_pct=sev,
                    confidence_score=conf,
                    timestamp_utc=obs_time
                )
            st.rerun()

    with p_col2:
        report = ledger.analyze_progression(curr_plant_id, curr_leaf_id)

        if report.observation_history:
            st.markdown("#### Disease Growth Rate & Alert Level")
            g1, g2, g3 = st.columns(3)

            rate_display = f"{report.progression_rate_daily:.2f} %/day" if report.progression_rate_daily is not None else "N/A"
            g1.metric("Growth Velocity (ΔS/day)", rate_display)
            g2.metric("Progression State", report.state.value)
            g3.metric("Risk Alert Level", report.risk_alert_level.value)

            # Trend Curve
            history_records = [
                {
                    "Timestamp": obs.timestamp_utc,
                    "Severity (%)": obs.severity_pct,
                    "Confidence (%)": obs.confidence_score
                }
                for obs in report.observation_history
            ]
            df_hist = pd.DataFrame(history_records)

            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=df_hist["Timestamp"],
                y=df_hist["Severity (%)"],
                mode="lines+markers",
                name="Severity (%)",
                line=dict(color="#dc2626", width=3),
                marker=dict(size=8)
            ))
            fig_trend.update_layout(
                title=f"Severity Trajectory: Plant {curr_plant_id} / Leaf {curr_leaf_id}",
                xaxis_title="UTC Timestamp",
                yaxis_title="Disease Severity Percentage (%)",
                height=350,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_trend, use_container_width=True)

            with st.expander("📋 View Raw Observation Log Table"):
                st.dataframe(df_hist, use_container_width=True)
        else:
            st.info(f"No historical records found for Plant {curr_plant_id} / Leaf {curr_leaf_id}.")


# -------------------------------------------------------------
# TAB 4: Benchmarking & LaTeX Export
# -------------------------------------------------------------
with tab4:
    st.subheader("📊 Experimental Benchmarking & Publication Table Exporter")
    st.caption("Compares Proposed FieldSight-Lite vs 4 literature baselines across clean and perturbed lighting regimes.")

    num_test_plants = st.slider("Number of Benchmark Plants (Strict Split)", min_value=4, max_value=16, value=8)

    if st.button("🚀 Run Comprehensive Benchmark Suite", type="primary"):
        with st.spinner("Generating Plant-ID split test cohort and evaluating baselines..."):
            dataset = create_synthetic_benchmark_dataset(num_plants=num_test_plants, leaves_per_plant=4, seed=42)
            _, _, test_set = split_by_plant_id(dataset, train_ratio=0.5, val_ratio=0.2, seed=42)

            bench_df = run_benchmark_suite(test_set, pipeline)
            st.session_state["bench_df"] = bench_df

    if "bench_df" in st.session_state:
        bench_df = st.session_state["bench_df"]

        st.markdown("### 🏆 Quantitative Comparison Table")
        st.dataframe(bench_df.style.highlight_max(subset=["Clean Leaf IoU (%)", "Perturbed Leaf IoU (%)", "Clean Lesion IoU (%)", "Perturbed Lesion IoU (%)"], color="#d1fae5")
                              .highlight_min(subset=["Clean Sev MAE (%)", "Perturbed Sev MAE (%)", "Robustness Drop (ΔMAE)"], color="#d1fae5"),
                     use_container_width=True)

        # Plotly comparison
        fig_bar = px.bar(
            bench_df,
            x="Method",
            y=["Clean Lesion IoU (%)", "Perturbed Lesion IoU (%)"],
            barmode="group",
            title="Lesion Segmentation IoU: Clean vs Perturbed Lighting",
            color_discrete_sequence=["#10b981", "#ef4444"]
        )
        fig_bar.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("### 📄 LaTeX Publication Code")
        latex_str = export_latex_table(bench_df, "data/benchmark_results/table.tex")
        st.code(latex_str, language="latex")

        d_col1, d_col2 = st.columns(2)
        with d_col1:
            st.download_button(
                label="📥 Download LaTeX Table (.tex)",
                data=latex_str,
                file_name="fieldsight_benchmarks.tex",
                mime="text/plain"
            )
        with d_col2:
            st.download_button(
                label="📥 Download CSV Summary (.csv)",
                data=bench_df.to_csv(index=False),
                file_name="fieldsight_benchmarks.csv",
                mime="text/csv"
            )
