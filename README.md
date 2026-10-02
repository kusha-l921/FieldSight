# FieldSight-Lite: Illumination-Robust, Training-Free Plant Disease Severity & Progression Monitoring on Edge Devices

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Edge Ready](https://img.shields.io/badge/Hardware-Edge%20%2F%20Raspberry%20Pi-brightgreen.svg)]()
[![Deterministic & Training-Free](https://img.shields.io/badge/Inference-Training--Free%20%2F%20Deterministic-blueviolet.svg)]()

> **FieldSight-Lite** is a deterministic, illumination-invariant, training-free computer vision platform designed to quantify plant leaf disease severity percentage ($S \in [0.0, 100.0]\%$) and track temporal growth velocity ($\Delta S / \text{day}$) on resource-constrained edge hardware.

---

## 🌟 Core System USPs & Scientific Integrity

1. **Illumination Decoupling via CIELAB & CLAHE:**
   Decouples spatial luminance from foliar chromaticity, applying localized CLAHE strictly to the $L^*$-channel to flatten harsh direct sunlight and cast shadows while preserving true leaf pigment coordinates.
2. **Two-Pass Dynamic Inlier MAD Anomaly Segmentation:**
   Self-calibrates healthy baseline tissue statistics per leaf via a two-pass Median and Median Absolute Deviation (MAD) model, isolating symptomatic chlorosis and necrosis without pre-training or neural network weights.
3. **Active Lighting Stress-Testing Suite:**
   Automatically subjects leaf imagery to 5 deterministic physical transformations (Shadow Ramps, Specular Glare, Overexposure, Underexposure, and Non-linear Gamma Shifts) to compute the **Lighting Robustness Score (LRS)** and **Mean Absolute Severity Drift (MASD)**.
4. **Temporal Disease Progression Velocity Tracking:**
   Maintains a SQLite ledger recording multi-day foliar observations, calculating continuous growth rates ($\Delta S / \text{day}$) and categorizing epidemiological trajectories (`STABLE`, `EMERGING`, `MODERATE`, `RAPID`).
5. **Scientific Integrity & Error Decoupling:**
   Optical or hardware failures (`NO_LEAF`, `BLURRY_IMAGE`, `EXCESSIVE_GLARE`) never evaluate to $0.0\%$ disease. Failures return typed error codes with downstream values set to `None`. Terminology strictly uses `NO_LESION_DETECTED` and `LESION_DETECTED`.

---

## 📐 Mathematical Formulations

### 1. Pre-Flight Optical Quality Gate
$$\text{Laplacian Variance: } \text{Var}(\nabla^2 I_{\text{gray}})$$
$$\text{Physical Lightness: } L_{\text{physical}} = \left(\frac{L_{\text{8bit}}}{255.0}\right) \times 100.0$$
$$\text{Specular Glare: } G = (L_{\text{physical}} \ge 96.0) \land (S_{\text{hsv}} \le 30)$$
$$\text{Deep Shadow: } D = (L_{\text{physical}} \le 10.0)$$

### 2. Two-Pass Robust Inlier MAD Segmentation
- **Pass 1:** Calculate initial medians $\tilde{\mu}_a, \tilde{\mu}_b$ and $\text{MAD}_a = 1.4826 \cdot \text{median}(|A_{\text{leaf}} - \tilde{\mu}_a|)$.
- **Healthy Reference Inliers:**
  $$H_{\text{ref}} = (A_{\text{leaf}} \le \tilde{\mu}_a + 1.5 \cdot \text{MAD}_a) \land (B_{\text{leaf}} \le \tilde{\mu}_b + 1.5 \cdot \text{MAD}_b)$$
- **Pass 2:** Recompute baseline parameters $\mu_a, \mu_b, \text{MAD}_a, \text{MAD}_b$ strictly on $H_{\text{ref}}$.
- **Chromatic Anomaly Distance:**
  $$D_{\text{chroma}}(x,y) = \sqrt{\left(\frac{a^*(x,y) - \mu_a}{\text{MAD}_a + \epsilon}\right)^2 + \left(\frac{b^*(x,y) - \mu_b}{\text{MAD}_b + \epsilon}\right)^2}$$
- **Lesion Pixel Decision:** $(D_{\text{chroma}} \ge k_{\text{thresh}}) \land (a^*(x,y) > \mu_a) \cap M_{\text{leaf}}$

### 3. Multi-Factor Reliability Confidence Score
$$C = 100.0 \times \left(w_q C_q + w_s C_s + w_t C_t + w_r C_r\right)$$
- $C_q = \text{clip}(1.0 - (G_{\text{ratio}} + D_{\text{ratio}}), 0.0, 1.0) \times \min\left(1.0, \frac{\text{Var}(\nabla^2 I)}{\tau_{\text{blur}}}\right)$
- $C_s = \exp\left(-\beta \frac{\text{Perimeter}(M_{\text{leaf}})}{\sqrt{\text{Area}(M_{\text{leaf}})}}\right)$
- $C_t = \text{clip}\left(\frac{\mu_{\text{lesion\_chroma}} - \mu_a}{3 \cdot (\text{MAD}_a + \text{MAD}_b) + \epsilon}, 0.0, 1.0\right)$
- $C_r = \frac{\text{LRS}}{100.0}$

### 4. Robustness & Temporal Progression Metrics
- **Mean Absolute Severity Drift (MASD):** $\text{MASD} = \frac{1}{K} \sum_{i=1}^K |S_i - S_0|$
- **Lighting Robustness Score (LRS):** $\text{LRS} = 100.0 \times \text{clip}\left(1.0 - \frac{\sigma(S_0, \dots, S_K)}{\mu(S_0, \dots, S_K) + \epsilon}, 0.0, 1.0\right)$
- **Progression Rate:** $R_{\text{prog}} = \frac{S(t_2) - S(t_1)}{\Delta t}$ (%/day)

---

## 📂 Repository Tree

```text
fieldsight_lite/
├── configs/
│   └── default_config.yaml        # All pipeline parameters and thresholds
├── src/
│   ├── schemas.py                 # Immutable dataclasses & runtime boundary validators
│   ├── provenance.py              # SHA-256 config hashing & RuntimeMetadata logger
│   ├── logger.py                  # Structured JSON/console logger with execution IDs
│   ├── profiler.py                # Latency (mean/median/P95), psutil RAM and CPU tracking
│   ├── camera.py                  # Thread-safe OpenCV capture with frame throttler
│   ├── quality.py                 # Pre-flight exposure, glare, and blur analysis
│   ├── preprocessing.py           # CIELAB conversion & localized L*-channel CLAHE
│   ├── leaf_segmentation.py       # Adaptive Otsu & spatial leaf extraction
│   ├── lesion_segmentation.py     # Two-pass robust MAD outlier chromatic anomaly detection
│   ├── morphology.py              # Connected components, hole filling, component filtering
│   ├── severity.py                # Severity & multi-factor confidence calculator
│   ├── robustness.py              # Perturbation suite, MASD, LRS, and FRI engine
│   ├── progression.py             # SQLite temporal ledger & trend classifier
│   └── pipeline.py                # Master FieldSightPipeline coordinator
├── experiments/
│   ├── dataset_loader.py          # PlantVillage synthetic/real loader with plant-ID split
│   ├── synthetic_perturb.py       # 5 deterministic physical lighting transformations
│   ├── baselines.py               # RGB, HSV, Global Otsu, & K-Means baselines
│   ├── evaluation.py              # IoU, Dice, MAE, RMSE calculator
│   └── run_benchmarks.py          # Automated benchmark runner & LaTeX table exporter
├── dashboard/
│   └── app.py                     # 4-Tab Streamlit research interface
├── cli/
│   └── main.py                    # Click CLI (single, batch, live, benchmark, progression)
├── tests/
│   ├── test_schemas.py
│   ├── test_quality.py
│   ├── test_preprocessing.py
│   ├── test_segmentation.py
│   ├── test_pipeline_integration.py
│   └── test_robustness.py
├── data/
│   └── temporal_db/
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 🚀 Quickstart & Installation

```bash
# 1. Clone repository and enter directory
cd /path/to/fieldsight_lite

# 2. Set up virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 💻 CLI Commands

### 1. Single Image Analysis
```bash
python3 cli/main.py image --input samples/leaf.jpg --output-dir outputs/ --stress-test --plant-id PLANT_01 --leaf-id LEAF_01
```

### 2. Batch Processing
```bash
python3 cli/main.py batch --input-dir samples/ --output-dir batch_results/ --export-csv
```

### 3. Live Edge Camera Monitoring
```bash
python3 cli/main.py camera --camera-index 0 --fps-limit 15
```

### 4. Automated Benchmarks vs Baselines & LaTeX Export
```bash
python3 cli/main.py benchmark --export-latex data/benchmark_results/table.tex --export-csv data/benchmark_results/summary.csv
```

### 5. Query Temporal Progression
```bash
python3 cli/main.py progression --plant-id PLANT_01 --leaf-id LEAF_01
```

---

## 🖥️ Streamlit Interactive Research Dashboard

Launch the 4-tab interactive research dashboard:

```bash
streamlit run dashboard/app.py
```

### Dashboard Tabs:
1. **🔍 1. Foliar Inspection:** Side-by-side Raw, $L^*_{\text{norm}}$, Leaf Mask, Lesion Mask, Heatmap Overlay, and KPI metric cards.
2. **⚡ 2. Active Lighting Stress Test:** Real-time 5-perturbation lighting stress test with mask stability IoUs, severity drift bar chart, and LRS gauge.
3. **📈 3. Temporal Progression Ledger:** Interactive Plotly trajectory curve, growth velocity ($\Delta S / \text{day}$), and risk state alerts.
4. **📊 4. Benchmarks & LaTeX Export:** Quantitative comparison vs baselines with one-click LaTeX code generation and CSV downloads.

---

## 🧪 Running the Test Suite

```bash
pytest tests/ -v
```
