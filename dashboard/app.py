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
import psutil
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
    page_title="FieldSight-Lite | Edge Foliar Diagnostics",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# High-Precision Monochrome HUD CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"], [data-testid="stAppViewContainer"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #09090b !important;
        color: #e4e4e7 !important;
    }

    [data-testid="stSidebar"] {
        background-color: #0d0d0e !important;
        border-right: 1px solid #27272a !important;
    }
    
    [data-testid="stHeader"] {
        background-color: rgba(9, 9, 11, 0.85) !important;
        backdrop-filter: blur(8px) !important;
    }

    code, pre, .mono-text {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Monochrome HUD Card Containers */
    .hud-card, .telemetry-card, .glass-card {
        background: #141416 !important;
        border: 1px solid #27272a !important;
        border-radius: 6px !important;
        padding: 16px 20px !important;
        box-shadow: inset 0 1px 0 0 rgba(255, 255, 255, 0.04), 0 4px 16px rgba(0, 0, 0, 0.4) !important;
        margin-bottom: 12px !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    }

    .hud-card:hover, .telemetry-card:hover, .glass-card:hover {
        border-color: #444444 !important;
        box-shadow: inset 0 1px 0 0 rgba(255, 255, 255, 0.08), 0 6px 20px rgba(0, 0, 0, 0.6) !important;
    }

    /* Monospace Metric & Telemetry Titles */
    .telemetry-title, .hud-title {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.70rem !important;
        font-weight: 600 !important;
        color: #a1a1aa !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
    }

    /* Numeric Readouts: Large Bold Pure White */
    .telemetry-value, .hud-value {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.85rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        margin-top: 4px !important;
        letter-spacing: -0.02em !important;
    }

    .telemetry-sub, .hud-sub {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.72rem !important;
        color: #71717a !important;
        margin-top: 2px !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }

    /* Zero Accent Colors: Clean Technical Tags */
    .hud-tag, .status-badge {
        display: inline-flex !important;
        align-items: center !important;
        gap: 6px !important;
        background: #202022 !important;
        color: #e4e4e7 !important;
        border: 1px solid #3f3f46 !important;
        border-radius: 4px !important;
        padding: 3px 8px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.68rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
    }

    .badge-emerald, .badge-amber, .badge-crimson, .badge-blue, .hud-tag-solid {
        background: #27272a !important;
        color: #ffffff !important;
        border: 1px solid #52525b !important;
    }
    
    .hud-tag-dim {
        background: #18181b !important;
        color: #a1a1aa !important;
        border: 1px solid #27272a !important;
    }

    /* Monochrome Edge Telemetry Pill */
    .edge-telemetry-pill {
        display: inline-flex !important;
        align-items: center !important;
        gap: 10px !important;
        background: #141416 !important;
        border: 1px solid #27272a !important;
        border-radius: 6px !important;
        padding: 6px 14px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.75rem !important;
        color: #d4d4d8 !important;
        letter-spacing: 0.04em !important;
    }

    .telemetry-dot {
        width: 6px !important;
        height: 6px !important;
        background-color: #ffffff !important;
        border-radius: 50% !important;
        box-shadow: 0 0 6px rgba(255, 255, 255, 0.8) !important;
        display: inline-block !important;
    }

    /* 5-Channel View Framing Cards */
    .frame-card {
        background: #111113 !important;
        border: 1px solid #2a2a2e !important;
        border-radius: 6px !important;
        padding: 8px !important;
        margin-bottom: 8px !important;
        transition: border-color 0.2s ease !important;
    }
    .frame-card:hover {
        border-color: #52525b !important;
    }
    .frame-caption {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.70rem !important;
        color: #a1a1aa !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        margin-top: 6px !important;
        text-align: center !important;
    }

    /* Streamlit Button Overrides */
    .stButton > button, .stDownloadButton > button {
        background-color: #1f1f23 !important;
        color: #ffffff !important;
        border: 1px solid #3f3f46 !important;
        border-radius: 6px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.80rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.04em !important;
        padding: 8px 18px !important;
        transition: all 0.15s ease-in-out !important;
    }
    
    .stButton > button:hover, .stDownloadButton > button:hover {
        background-color: #ffffff !important;
        color: #000000 !important;
        border-color: #ffffff !important;
        box-shadow: 0 0 12px rgba(255, 255, 255, 0.25) !important;
    }

    .stButton > button:active, .stDownloadButton > button:active {
        background-color: #e4e4e7 !important;
        color: #000000 !important;
    }

    /* Streamlit Tabs Overrides */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px !important;
        background-color: #111113 !important;
        border: 1px solid #27272a !important;
        border-radius: 6px !important;
        padding: 4px !important;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 4px !important;
        color: #71717a !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.04em !important;
        padding: 8px 16px !important;
        border: 1px solid transparent !important;
        background-color: transparent !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: #202023 !important;
        color: #ffffff !important;
        border: 1px solid #3f3f46 !important;
        font-weight: 700 !important;
    }

    /* Input & Selection Widgets */
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
        background-color: #141416 !important;
        border-color: #27272a !important;
        color: #ffffff !important;
        border-radius: 6px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.82rem !important;
    }

    .stRadio label, .stCheckbox label {
        color: #d4d4d8 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.80rem !important;
    }

    /* Sliders */
    .stSlider [data-baseweb="slider"] {
        accent-color: #ffffff !important;
    }

    /* Streamlit Metric Overrides */
    [data-testid="stMetric"] {
        background: #141416 !important;
        border: 1px solid #27272a !important;
        border-radius: 6px !important;
        padding: 12px 16px !important;
    }
    [data-testid="stMetricLabel"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.70rem !important;
        color: #a1a1aa !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
    }
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.55rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
    }

    /* Expander */
    [data-testid="stExpander"] {
        background: #141416 !important;
        border: 1px solid #27272a !important;
        border-radius: 6px !important;
    }
    [data-testid="stExpander"] summary {
        font-family: 'JetBrains Mono', monospace !important;
        color: #d4d4d8 !important;
        font-size: 0.82rem !important;
    }

    /* Streamlit Alert Boxes */
    .stAlert {
        background: #141416 !important;
        border: 1px solid #383838 !important;
        border-radius: 6px !important;
        color: #e4e4e7 !important;
    }
    .stAlert [data-testid="stMarkdownContainer"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.80rem !important;
    }

    /* Dividers */
    hr {
        border-color: #27272a !important;
        margin: 18px 0 !important;
    }
</style>
""", unsafe_allow_html=True)


def create_hud_overlay(
    original_bgr: np.ndarray,
    lesion_mask: Optional[np.ndarray],
    mode: str = "HUD Wireframe (Monochrome)"
) -> np.ndarray:
    """
    Renders high-precision monochrome HUD overlay.
    Desaturates foliage into technical grayscale and outlines quantified
    lesions with high-contrast white wireframe/contour and luminance fill.
    """
    if original_bgr is None:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
    mono_base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if lesion_mask is None or np.count_nonzero(lesion_mask == 255) == 0:
        return mono_base

    overlay = mono_base.copy()
    mask_indices = (lesion_mask == 255)

    if "Monochrome" in mode:
        # High-contrast white translucent highlight
        white_highlight = np.full_like(mono_base, 255)
        blended = cv2.addWeighted(mono_base, 0.45, white_highlight, 0.55, 0.0)
        overlay[mask_indices] = blended[mask_indices]

        # Draw crisp white boundary contours
        contours, _ = cv2.findContours(lesion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)
    else:
        # Fallback colored highlight
        red_color = np.array([0, 0, 240], dtype=np.uint8)
        blended = cv2.addWeighted(original_bgr, 0.55, np.full_like(original_bgr, red_color), 0.45, 0.0)
        overlay = original_bgr.copy()
        overlay[mask_indices] = blended[mask_indices]
        contours, _ = cv2.findContours(lesion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 1)

    return overlay


def apply_monochrome_plotly_theme(fig: go.Figure, title: str = "", height: int = 350) -> go.Figure:
    """
    Reskins Plotly figures into technical monochrome HUD styling.
    """
    fig.update_layout(
        title=dict(
            text=f"<b>{title.upper()}</b>" if title else "",
            font=dict(family="JetBrains Mono, monospace", size=12, color="#ffffff"),
            x=0.01,
            y=0.96
        ),
        paper_bgcolor="#121214",
        plot_bgcolor="#121214",
        height=height,
        margin=dict(l=35, r=20, t=50, b=35),
        font=dict(family="JetBrains Mono, monospace", color="#a1a1aa", size=11),
        xaxis=dict(
            gridcolor="#222225",
            zerolinecolor="#2e2e33",
            linecolor="#2e2e33",
            tickfont=dict(family="JetBrains Mono, monospace", color="#a1a1aa", size=10),
            title_font=dict(family="JetBrains Mono, monospace", color="#d4d4d8", size=11)
        ),
        yaxis=dict(
            gridcolor="#222225",
            zerolinecolor="#2e2e33",
            linecolor="#2e2e33",
            tickfont=dict(family="JetBrains Mono, monospace", color="#a1a1aa", size=10),
            title_font=dict(family="JetBrains Mono, monospace", color="#d4d4d8", size=11)
        ),
        legend=dict(
            font=dict(family="JetBrains Mono, monospace", color="#d4d4d8", size=10),
            bgcolor="rgba(18, 18, 20, 0.9)",
            bordercolor="#27272a",
            borderwidth=1
        ),
        hoverlabel=dict(
            bgcolor="#18181b",
            bordercolor="#3f3f46",
            font=dict(family="JetBrains Mono, monospace", color="#ffffff", size=11)
        )
    )
    return fig


@st.cache_resource
def get_pipeline(config_path: str = "configs/default_config.yaml") -> FieldSightPipeline:
    return FieldSightPipeline(config_path)


@st.cache_resource
def get_ledger(db_path: str = "data/temporal_db/field_history.db") -> ProgressionLedger:
    return ProgressionLedger(db_path=db_path)


# Sidebar Configuration
st.sidebar.markdown("""
<div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
    <div style="font-size: 1.4rem; color: #ffffff;">◈</div>
    <div>
        <div style="font-size: 1.05rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; letter-spacing: -0.01em; color: #ffffff;">FIELDSIGHT-LITE</div>
        <div style="font-size: 0.68rem; color: #a1a1aa; font-family: 'JetBrains Mono', monospace; text-transform: uppercase; letter-spacing: 0.08em;">EDGE PRECISION VISION</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.caption("Illumination-Invariant Foliar Disease Quantification & Growth Velocity Tracker")
st.sidebar.markdown("---")

config_path = st.sidebar.text_input("Config Path", "configs/default_config.yaml")
db_path = st.sidebar.text_input("SQLite DB Path", "data/temporal_db/field_history.db")

pipeline = get_pipeline(config_path)
ledger = get_ledger(db_path)

# System Resource Telemetry in Sidebar
mem_info = psutil.virtual_memory()
cpu_pct = psutil.cpu_percent(interval=None)

st.sidebar.markdown("### ◈ EDGE TELEMETRY")
st.sidebar.markdown(f"""
<div class="hud-card">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
        <span class="telemetry-title">Execution Mode</span>
        <span class="hud-tag"><span class="telemetry-dot"></span> CPU Headless</span>
    </div>
    <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 6px;">
        <span style="font-size: 0.78rem; font-family: 'JetBrains Mono', monospace; color: #a1a1aa;">MEMORY RAM:</span>
        <span style="font-size: 0.80rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; color: #ffffff;">{mem_info.percent:.1f}% ({mem_info.used / (1024**2):.0f} MB)</span>
    </div>
    <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 4px;">
        <span style="font-size: 0.78rem; font-family: 'JetBrains Mono', monospace; color: #a1a1aa;">CPU LOAD:</span>
        <span style="font-size: 0.80rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; color: #ffffff;">{cpu_pct:.1f}%</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Main Title Header with Telemetry Pill
header_col, pill_col = st.columns([3, 2])
with header_col:
    st.title("◈ FieldSight-Lite Console")
    st.caption("Deterministic, Training-Free Plant Disease Severity & Progression Quantification on Edge Devices")

with pill_col:
    st.markdown(f"""
    <div style="text-align: right; margin-top: 14px;">
        <div class="edge-telemetry-pill">
            <span><span class="telemetry-dot"></span> ENGINE ONLINE</span>
            <span style="color: #3f3f46;">|</span>
            <span>TARGET: <strong>EDGE ARM/X86</strong></span>
            <span style="color: #3f3f46;">|</span>
            <span>RAM: <strong>{mem_info.percent:.0f}%</strong></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Tab Navigation
tab1, tab2, tab3, tab4 = st.tabs([
    "01 // FOLIAR INSPECTION",
    "02 // OPTICAL STRESS TEST",
    "03 // TEMPORAL LEDGER",
    "04 // BENCHMARKS & LATEX"
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
                    "Critical High-Severity Blight (53.4% Severity)",
                    "Natural Background In-The-Wild (samples/test_leaf.jpg)",
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
            elif "Critical High-Severity" in preset_choice:
                # Generate high severity leaf (>50% necrosis)
                h, w = 400, 400
                img = np.ones((h, w, 3), dtype=np.uint8) * 210
                cv2.ellipse(img, (200, 200), (120, 80), 0, 0, 360, (35, 140, 60), -1)
                cv2.ellipse(img, (220, 200), (85, 60), 0, 0, 360, (30, 70, 160), -1)
                image_bgr = img
            elif "In-The-Wild" in preset_choice:
                if os.path.exists("samples/test_leaf.jpg"):
                    image_bgr = cv2.imread("samples/test_leaf.jpg")
                else:
                    img, _, _, _ = generate_synthetic_leaf(severity_target_pct=15.0, seed=707)
                    image_bgr = img
            elif "Glare" in preset_choice:
                img, _, _, _ = generate_synthetic_leaf(severity_target_pct=10.0, seed=505)
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

        # Check for unsegmented full-canvas warning flag
        if any(err.code.value == "SEGMENTATION_FAILURE" for err in res.errors):
            st.warning("◈ CANOPY SEGMENTATION ALERT: Leaf mask spans >98% of the image canvas. Background may be partially unsegmented.")

        st.markdown("---")

        # KPI Telemetry Cards with Monochrome HUD Tags
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

        with kpi1:
            if res.severity is not None:
                sev_val = res.severity.severity_pct
                badge_text = "MILD" if sev_val < 5.0 else ("MODERATE" if sev_val <= 20.0 else "SEVERE")

                st.markdown(f"""
                <div class="hud-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span class="telemetry-title">Severity</span>
                        <span class="hud-tag">[{badge_text}]</span>
                    </div>
                    <div class="telemetry-value">{sev_val:.2f}%</div>
                    <div class="telemetry-sub">Lesion / Canopy Lamina</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="hud-card">
                    <div class="telemetry-title">Severity</div>
                    <div class="telemetry-value" style="color: #52525b;">N/A</div>
                    <div class="telemetry-sub">Quality Gate Rejected</div>
                </div>
                """, unsafe_allow_html=True)

        with kpi2:
            if res.severity is not None:
                st.markdown(f"""
                <div class="hud-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span class="telemetry-title">Healthy Tissue</span>
                        <span class="hud-tag">CHLOROPHYLL</span>
                    </div>
                    <div class="telemetry-value">{res.severity.unaffected_pct:.2f}%</div>
                    <div class="telemetry-sub">Surviving Green Lamina</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="hud-card">
                    <div class="telemetry-title">Healthy Tissue</div>
                    <div class="telemetry-value" style="color: #52525b;">N/A</div>
                    <div class="telemetry-sub">Quality Gate Rejected</div>
                </div>
                """, unsafe_allow_html=True)

        with kpi3:
            q_status = res.quality.status.value
            st.markdown(f"""
            <div class="hud-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span class="telemetry-title">Quality Gate</span>
                    <span class="hud-tag">[{q_status}]</span>
                </div>
                <div class="telemetry-value" style="font-size: 1.55rem; margin-top: 6px;">VAR: {res.quality.laplacian_variance:.0f}</div>
                <div class="telemetry-sub">MEAN L*: {res.quality.mean_lightness:.1f}</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi4:
            conf_val = res.severity.confidence_score if res.severity is not None else 0.0
            st.markdown(f"""
            <div class="hud-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span class="telemetry-title">Confidence</span>
                    <span class="hud-tag">MULTI-FACTOR</span>
                </div>
                <div class="telemetry-value">{conf_val:.1f}%</div>
                <div class="telemetry-sub">Reliability Index</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi5:
            lat_ms = res.performance.latency_mean_ms
            st.markdown(f"""
            <div class="hud-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span class="telemetry-title">Edge Latency</span>
                    <span class="hud-tag">{res.performance.fps:.1f} FPS</span>
                </div>
                <div class="telemetry-value" style="font-size: 1.55rem; margin-top: 6px;">{lat_ms:.1f} MS</div>
                <div class="telemetry-sub">Single Core CPU</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # 5-Channel Visual Decomposition Panel
        st.markdown("### ◈ 5-CHANNEL OPTICAL DECOMPOSITION")

        ctrl_col1, ctrl_col2 = st.columns([3, 2])
        with ctrl_col1:
            view_mode = st.radio(
                "Decomposition Layout",
                ["Multi-Panel Grid", "Tabbed Lightbox Inspector"],
                horizontal=True
            )
        with ctrl_col2:
            overlay_style = st.radio(
                "Channel 05 Overlay Style",
                ["HUD Wireframe (Monochrome)", "Chromatic Red (Legacy)"],
                horizontal=True
            )

        hud_overlay = create_hud_overlay(
            original_bgr=image_bgr,
            lesion_mask=res.segmentation.lesion_mask if res.segmentation else None,
            mode=overlay_style
        )

        if view_mode == "Multi-Panel Grid":
            v_col1, v_col2, v_col3, v_col4, v_col5 = st.columns(5)

            # 1. Raw BGR
            with v_col1:
                st.markdown('<div class="frame-card">', unsafe_allow_html=True)
                st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
                st.markdown('<div class="frame-caption">01_RAW_INPUT</div></div>', unsafe_allow_html=True)

            # 2. L* Lightness View
            with v_col2:
                st.markdown('<div class="frame-card">', unsafe_allow_html=True)
                if res.quality.is_valid_for_processing:
                    _, l_norm, _ = pipeline.normalizer.normalize(image_bgr)
                    st.image(l_norm, clamp=True)
                else:
                    st.info("Quality gate failed")
                st.markdown('<div class="frame-caption">02_CLAHE_L_CHANNEL</div></div>', unsafe_allow_html=True)

            # 3. Leaf Mask
            with v_col3:
                st.markdown('<div class="frame-card">', unsafe_allow_html=True)
                if res.segmentation is not None:
                    st.image(res.segmentation.leaf_mask)
                else:
                    st.warning("No Leaf Mask")
                st.markdown('<div class="frame-caption">03_CANOPY_MASK</div></div>', unsafe_allow_html=True)

            # 4. Lesion Mask
            with v_col4:
                st.markdown('<div class="frame-card">', unsafe_allow_html=True)
                if res.segmentation is not None:
                    st.image(res.segmentation.lesion_mask)
                else:
                    st.warning("No Lesion Mask")
                st.markdown('<div class="frame-caption">04_LESION_MASK</div></div>', unsafe_allow_html=True)

            # 5. Overlay View
            with v_col5:
                st.markdown('<div class="frame-card">', unsafe_allow_html=True)
                if res.segmentation is not None:
                    st.image(cv2.cvtColor(hud_overlay, cv2.COLOR_BGR2RGB))
                else:
                    st.warning("No Overlay")
                st.markdown('<div class="frame-caption">05_SEGMENTATION_OVERLAY</div></div>', unsafe_allow_html=True)

        else:
            c_tab1, c_tab2, c_tab3, c_tab4, c_tab5 = st.tabs([
                "01 // RAW INPUT",
                "02 // CLAHE L* CHANNEL",
                "03 // CANOPY MASK",
                "04 // LESION MASK",
                "05 // SEGMENTATION OVERLAY"
            ])
            with c_tab1:
                st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), caption="01_RAW_INPUT: Original Unprocessed Optical Acquisition")
            with c_tab2:
                if res.quality.is_valid_for_processing:
                    _, l_norm, _ = pipeline.normalizer.normalize(image_bgr)
                    st.image(l_norm, caption="02_CLAHE_L_CHANNEL: Illumination Decoupled L*-Channel Normalization", clamp=True)
                else:
                    st.info("Quality gate failed")
            with c_tab3:
                if res.segmentation is not None:
                    st.image(res.segmentation.leaf_mask, caption=f"03_CANOPY_MASK: Segmented Foliar Blade Area: {res.segmentation.leaf_area_px:,} px")
            with c_tab4:
                if res.segmentation is not None:
                    st.image(res.segmentation.lesion_mask, caption=f"04_LESION_MASK: Quantified Lesion Footprint: {res.segmentation.lesion_area_px:,} px")
            with c_tab5:
                st.image(cv2.cvtColor(hud_overlay, cv2.COLOR_BGR2RGB), caption="05_SEGMENTATION_OVERLAY: High-Precision HUD Diagnostic Wireframe")

        # Baseline Inlier Statistics
        if res.segmentation is not None:
            with st.expander("◈ DYNAMIC CIELAB STATISTICAL CALIBRATION (PASS 2 ANCHORED BASELINE)", expanded=False):
                s_c1, s_c2, s_c3, s_c4 = st.columns(4)
                s_c1.metric("Healthy Median a*", f"{res.segmentation.healthy_mu_a:.2f}")
                s_c2.metric("Healthy MAD a*", f"{res.segmentation.healthy_mad_a:.2f}")
                s_c3.metric("Healthy Median b*", f"{res.segmentation.healthy_mu_b:.2f}")
                s_c4.metric("Healthy MAD b*", f"{res.segmentation.healthy_mad_b:.2f}")

        # JSON Export Download
        json_dict = pipeline.export_result_dict(res)
        st.download_button(
            label="DOWNLOAD STRUCTURED JSON RESULT (RESULT.JSON)",
            data=pd.Series(json_dict).to_json(indent=2),
            file_name=f"{p_id}_{l_id}_result.json",
            mime="application/json"
        )


# -------------------------------------------------------------
# TAB 2: Active Lighting Stress Test
# -------------------------------------------------------------
with tab2:
    st.subheader("ACTIVE OPTICAL LIGHTING STRESS-TESTING ENGINE")
    st.caption("Applies 5 deterministic physical transformations to quantify Lighting Robustness Score (LRS) and Severity Drift (MASD).")

    if image_bgr is None:
        st.info("Please select or upload an image in Tab 1 first.")
    else:
        if st.button("RUN ACTIVE 5-PERTURBATION STRESS SUITE", type="primary"):
            with st.spinner("Subjecting leaf to physical illumination stress..."):
                stress_res = pipeline.process_image(
                    image_input=image_bgr,
                    run_stress_test=True
                )

            if stress_res.robustness is not None:
                r_rep = stress_res.robustness

                st.markdown("### ◈ ROBUSTNESS SCORECARD")
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("Lighting Robustness Score (LRS)", f"{r_rep.lighting_robustness_score:.1f}%")
                r2.metric("Mean Absolute Severity Drift (MASD)", f"{r_rep.mean_severity_drift:.2f}%")
                r3.metric("Severity Std Dev (σ)", f"{r_rep.severity_std_dev:.2f}%")
                r4.metric("Field Robustness Index (FRI)", f"{r_rep.field_robustness_index:.1f}%")

                st.markdown("---")
                st.markdown("### ◈ PERTURBATION BREAKDOWN & MASK STABILITY")

                p_cols = st.columns(len(r_rep.perturbation_breakdown))
                perts = generate_all_perturbations(image_bgr)

                drift_data = [{"Condition": "Baseline (S0)", "Severity (%)": r_rep.original_severity}]

                for idx, p in enumerate(r_rep.perturbation_breakdown):
                    with p_cols[idx]:
                        pert_img = perts.get(p.perturbation_name)
                        if pert_img is not None:
                            st.markdown('<div class="frame-card">', unsafe_allow_html=True)
                            st.image(cv2.cvtColor(pert_img, cv2.COLOR_BGR2RGB))
                            st.markdown(f'<div class="frame-caption">{p.perturbation_name}</div></div>', unsafe_allow_html=True)
                        st.markdown(f"**SEVERITY:** `{p.perturbed_severity_pct:.2f}%`")
                        st.markdown(f"**DRIFT:** `Δ {p.severity_drift_delta:.2f}%`")
                        st.markdown(f"**LESION IOU:** `{p.lesion_mask_stability_iou * 100.0:.1f}%`")

                    drift_data.append({
                        "Condition": p.perturbation_name,
                        "Severity (%)": p.perturbed_severity_pct
                    })

                # Plotly Monochrome Drift Chart
                df_drift = pd.DataFrame(drift_data)
                fig_drift = go.Figure()
                fig_drift.add_trace(go.Bar(
                    x=df_drift["Condition"],
                    y=df_drift["Severity (%)"],
                    marker=dict(
                        color="#ffffff",
                        line=dict(color="#71717a", width=1)
                    ),
                    name="Severity (%)"
                ))
                apply_monochrome_plotly_theme(fig_drift, title="Severity Percentage across Environmental Lighting Variations", height=340)
                st.plotly_chart(fig_drift)


# -------------------------------------------------------------
# TAB 3: Temporal Progression Ledger
# -------------------------------------------------------------
with tab3:
    st.subheader("LONGITUDINAL PROGRESSION LEDGER & GROWTH VELOCITY")
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

        if st.button("POPULATE 14-DAY FIELD PROGRESSION"):
            base_time = datetime.now(timezone.utc) - timedelta(days=14)
            for day in range(15):
                obs_time = (base_time + timedelta(days=day)).strftime("%Y-%m-%dT%H:%M:%SZ")
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
                line=dict(color="#ffffff", width=2),
                marker=dict(size=6, color="#ffffff", line=dict(color="#71717a", width=1)),
                fill='tozeroy',
                fillcolor='rgba(255, 255, 255, 0.05)'
            ))
            apply_monochrome_plotly_theme(fig_trend, title=f"Severity Trajectory: Plant {curr_plant_id} / Leaf {curr_leaf_id}", height=340)
            st.plotly_chart(fig_trend)

            with st.expander("◈ VIEW RAW OBSERVATION LOG TABLE"):
                st.dataframe(df_hist)
        else:
            st.info(f"No historical records found for Plant {curr_plant_id} / Leaf {curr_leaf_id}.")


# -------------------------------------------------------------
# TAB 4: Benchmarking & LaTeX Export
# -------------------------------------------------------------
with tab4:
    st.subheader("EXPERIMENTAL BENCHMARKING & PUBLICATION TABLE EXPORTER")
    st.caption("Compares Proposed FieldSight-Lite vs 4 literature baselines across clean and perturbed lighting regimes.")

    bench_source = st.radio(
        "Benchmark Evaluation Source",
        [
            "[01] Primary Empirical Ground-Truth Benchmark (N=30 Real Leaves)",
            "[02] Interactive Live Evaluation (Synthetic / Dynamic Split)"
        ],
        horizontal=False
    )

    real_tex_path = "results/table1_real_benchmarks.tex"
    real_csv_path = "results/table1_real_benchmarks.csv"

    if "Primary Empirical" in bench_source:
        st.info("Displaying verified publication results evaluated on N=30 human-annotated leaf images across Mild (<5%), Moderate (5-20%), and Severe (>20%) tiers.")

        if os.path.exists(real_csv_path):
            bench_df = pd.read_csv(real_csv_path)
            st.markdown("### ◈ EMPIRICAL BENCHMARK RESULTS (TABLE 1)")
            st.dataframe(
                bench_df.style.highlight_max(subset=["Clean Leaf IoU (%)", "Perturbed Leaf IoU (%)", "Clean Lesion IoU (%)", "Perturbed Lesion IoU (%)"], color="#27272a")
                              .highlight_min(subset=["Clean Sev MAE (%)", "Perturbed Sev MAE (%)", "Robustness Drop (ΔMAE)"], color="#27272a")
            )

            # Monochrome Bar chart
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=bench_df["Method"],
                y=bench_df["Clean Lesion IoU (%)"],
                name="Clean Lesion IoU (%)",
                marker=dict(color="#ffffff", line=dict(color="#a1a1aa", width=1))
            ))
            fig_bar.add_trace(go.Bar(
                x=bench_df["Method"],
                y=bench_df["Perturbed Lesion IoU (%)"],
                name="Perturbed Lesion IoU (%)",
                marker=dict(color="#52525b", line=dict(color="#71717a", width=1))
            ))
            fig_bar.update_layout(barmode="group")
            apply_monochrome_plotly_theme(fig_bar, title="Empirical Lesion IoU: Clean vs Perturbed Lighting (N=30 Real Leaves)", height=380)
            st.plotly_chart(fig_bar)

            if os.path.exists(real_tex_path):
                with open(real_tex_path, "r", encoding="utf-8") as f:
                    latex_str = f.read()
                st.markdown("### ◈ LATEX PUBLICATION CODE (`results/table1_real_benchmarks.tex`)")
                st.code(latex_str, language="latex")

                d_col1, d_col2 = st.columns(2)
                with d_col1:
                    st.download_button(
                        label="DOWNLOAD TABLE 1 LATEX (.TEX)",
                        data=latex_str,
                        file_name="table1_real_benchmarks.tex",
                        mime="text/plain"
                    )
                with d_col2:
                    st.download_button(
                        label="DOWNLOAD TABLE 1 CSV (.CSV)",
                        data=bench_df.to_csv(index=False),
                        file_name="table1_real_benchmarks.csv",
                        mime="text/csv"
                    )
        else:
            st.warning("`results/table1_real_benchmarks.csv` not found. Please run `python cli/main.py benchmark` to generate it.")
    else:
        st.markdown("#### ◈ INTERACTIVE BENCHMARK (SYNTHETIC COHORT)")
        num_test_plants = st.slider("Number of Benchmark Plants (Strict Split)", min_value=4, max_value=16, value=8)

        if st.button("RUN INTERACTIVE BENCHMARK SUITE", type="primary"):
            with st.spinner("Generating Plant-ID split test cohort and evaluating baselines..."):
                dataset = create_synthetic_benchmark_dataset(num_plants=num_test_plants, leaves_per_plant=4, seed=42)
                _, _, test_set = split_by_plant_id(dataset, train_ratio=0.5, val_ratio=0.2, seed=42)

                bench_df = run_benchmark_suite(test_set, pipeline)
                st.session_state["interactive_bench_df"] = bench_df

        if "interactive_bench_df" in st.session_state:
            bench_df = st.session_state["interactive_bench_df"]

            st.markdown("### ◈ INTERACTIVE COMPARISON TABLE")
            st.dataframe(bench_df.style.highlight_max(subset=["Clean Leaf IoU (%)", "Perturbed Leaf IoU (%)", "Clean Lesion IoU (%)", "Perturbed Lesion IoU (%)"], color="#27272a")
                                  .highlight_min(subset=["Clean Sev MAE (%)", "Perturbed Sev MAE (%)", "Robustness Drop (ΔMAE)"], color="#27272a"))

            # Plotly comparison
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=bench_df["Method"],
                y=bench_df["Clean Lesion IoU (%)"],
                name="Clean Lesion IoU (%)",
                marker=dict(color="#ffffff", line=dict(color="#a1a1aa", width=1))
            ))
            fig_bar.add_trace(go.Bar(
                x=bench_df["Method"],
                y=bench_df["Perturbed Lesion IoU (%)"],
                name="Perturbed Lesion IoU (%)",
                marker=dict(color="#52525b", line=dict(color="#71717a", width=1))
            ))
            fig_bar.update_layout(barmode="group")
            apply_monochrome_plotly_theme(fig_bar, title="Interactive Lesion IoU: Clean vs Perturbed Lighting", height=380)
            st.plotly_chart(fig_bar)

            st.markdown("### ◈ LATEX PUBLICATION CODE")
            latex_str = export_latex_table(bench_df, "data/benchmark_results/table.tex")
            st.code(latex_str, language="latex")

            d_col1, d_col2 = st.columns(2)
            with d_col1:
                st.download_button(
                    label="DOWNLOAD LATEX TABLE (.TEX)",
                    data=latex_str,
                    file_name="fieldsight_benchmarks.tex",
                    mime="text/plain"
                )
            with d_col2:
                st.download_button(
                    label="DOWNLOAD CSV SUMMARY (.CSV)",
                    data=bench_df.to_csv(index=False),
                    file_name="fieldsight_benchmarks.csv",
                    mime="text/csv"
                )

