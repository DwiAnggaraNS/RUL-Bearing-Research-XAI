"""
4_FINAL_SUMMARY.py
==================
Dataset Transparency Profiling (B1 vs B2) + Comprehensive Multi-Architecture
RUL Model Comparison Across Feature Domain Scenarios (S1/S2/S3).

Outputs
-------
  b1_vs_b2_descriptive_stats.csv
  plot0_feature_distribution.png

  table1_domain_comparison.csv          <- Multi-level: 4 models x 3 scenarios x 4 metrics
  table2_s3_vs_s4.csv
  table3_shap_consistency.csv

  plot1_rmse_smape_grouped_bar.png      <- RMSE & sMAPE comparison bar chart
  plot2_bhi_curves_per_scenario.png     <- All 4 models per scenario (3 subplots)
  plot3_tail_rmse_countdown.png         <- Last TAIL_N steps countdown
  plot4_radar_metrics.png               <- NEW: Radar chart - holistic model comparison
  plot5_error_distribution.png          <- NEW: Violin/box error distribution per model & scenario
  plot6_prediction_scatter.png          <- NEW: Pred vs True scatter for S3 best region
  plot7_rolling_rmse.png               <- NEW: Rolling-window RMSE over time
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines   import Line2D
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math

# =============================================================================
# CONFIGURATION
# =============================================================================

TRAIN_DATA_PATH = r"D:\ProyekDosen\Riset Bearing ShaftMissalignment + XAI\bearing_1\processed_bearing1.parquet"
TEST_DATA_PATH  = r"D:\ProyekDosen\Riset Bearing ShaftMissalignment + XAI\bearing_2\processed_bearing2.parquet"
RESULTS_DIR     = r"D:\ProyekDosen\Riset Bearing ShaftMissalignment + XAI\results_LSTM"
OUTPUT_DIR      = RESULTS_DIR

MODELS    = ["LSTM", "GRU", "TCN", "CNNLSTM"]
SCENARIOS = ["S1_TimeDomain", "S2_FreqDomain", "S3_Combined"]

KEY_FEATURES = [
    'td_rms_y', 'td_kurtosis_y', 'td_peak_factor_y',
    'fd_bpfi_energy_y', 'fd_spectral_energy_y', 'td_p2p_y'
]

BEARING_LIFE_S = 392_275
TAIL_N         = 50      # countdown steps for Plot 3

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# GLOBAL PLOT STYLE  (Times New Roman, journal-ready, white bg)
# =============================================================================

plt.style.use('default')
plt.rcParams.update({
    'font.family'           : 'serif',
    'font.serif'            : ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size'             : 11,
    'axes.labelsize'        : 12,
    'axes.titlesize'        : 13,
    'xtick.labelsize'       : 10,
    'ytick.labelsize'       : 10,
    'legend.fontsize'       : 10,
    'legend.title_fontsize' : 11,
    'figure.facecolor'      : 'white',
    'axes.facecolor'        : 'white',
    'savefig.facecolor'     : 'white',
    'axes.edgecolor'        : 'black',
    'axes.grid'             : True,
    'axes.grid.which'       : 'both',
    'grid.color'            : '#c0c0c0',
    'grid.alpha'            : 0.35,
    'grid.linestyle'        : '--',
    'grid.linewidth'        : 0.6,
    'xtick.direction'       : 'in',
    'ytick.direction'       : 'in',
    'xtick.top'             : True,
    'ytick.right'           : True,
    'xtick.minor.visible'   : True,
    'ytick.minor.visible'   : True,
    'legend.frameon'        : True,
    'legend.framealpha'     : 0.95,
    'legend.edgecolor'      : 'black',
    'legend.fancybox'       : False,
    'lines.linewidth'       : 1.8,
})

# Color-blind-friendly palette (Wong 2011)
PALETTE = {
    'LSTM'    : '#0072B2',   # Blue
    'GRU'     : '#D55E00',   # Vermilion
    'TCN'     : '#009E73',   # Bluish-green
    'CNNLSTM' : '#CC79A7',   # Reddish-purple
    'B1'      : '#0072B2',
    'B2'      : '#D55E00',
    'true'    : '#1a1a1a',   # Near-black for actual signal
    'S1'      : '#0072B2',
    'S2'      : '#D55E00',
    'S3'      : '#009E73',
}

# Distinct linestyles for 4 models on same subplot
MODEL_LINESTYLE = {
    'LSTM'    : '-',
    'GRU'     : '--',
    'TCN'     : '-.',
    'CNNLSTM' : ':',
}

SCENARIO_LABEL = {
    'S1_TimeDomain' : 'S1 (Time)',
    'S2_FreqDomain' : 'S2 (Freq)',
    'S3_Combined'   : 'S3 (Combined)',
}

SCENARIO_TITLE = {
    'S1_TimeDomain' : 'Scenario 1 — Time Domain Features',
    'S2_FreqDomain' : 'Scenario 2 — Frequency Domain Features',
    'S3_Combined'   : 'Scenario 3 — Combined Features',
}


def apply_journal_style(ax):
    ax.set_facecolor('white')
    ax.minorticks_on()
    ax.grid(True, which='major', linestyle='--', linewidth=0.7, alpha=0.35, color='#b0b0b0')
    ax.grid(True, which='minor', linestyle=':',  linewidth=0.5, alpha=0.20, color='#d0d0d0')


def save_fig(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {path}")


# =============================================================================
# PART 1 — DATASET TRANSPARENCY (B1 vs B2)
# =============================================================================

def load_raw_parquets():
    print("\n[PART 1] Loading raw parquet files...")
    df_b1 = pd.read_parquet(TRAIN_DATA_PATH)
    df_b2 = pd.read_parquet(TEST_DATA_PATH)
    print(f"  B1 shape: {df_b1.shape}  |  B2 shape: {df_b2.shape}")
    return df_b1, df_b2


def build_descriptive_stats(df_b1: pd.DataFrame, df_b2: pd.DataFrame) -> pd.DataFrame:
    """Mean, Std, Min, Max for KEY_FEATURES — side-by-side B1 vs B2."""
    available = [f for f in KEY_FEATURES if f in df_b1.columns]
    missing   = set(KEY_FEATURES) - set(available)
    if missing:
        print(f"  [WARN] Features not found in dataset: {missing}")

    rows = []
    for feat in available:
        for bearing, df in [("B1", df_b1), ("B2", df_b2)]:
            s = df[feat].dropna()
            rows.append({
                "Feature" : feat,
                "Bearing" : bearing,
                "Mean"    : s.mean(),
                "Std"     : s.std(),
                "Min"     : s.min(),
                "Max"     : s.max(),
            })

    stats = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, "b1_vs_b2_descriptive_stats.csv")
    stats.to_csv(path, index=False)
    print(f"[SAVED] b1_vs_b2_descriptive_stats.csv")
    return stats, available


def plot_feature_distribution(df_b1: pd.DataFrame, df_b2: pd.DataFrame, features: list):
    print("\n[OK] Generating Plot 0: Feature Distribution (KDE)...")
    n    = len(features)
    cols = 3
    rows = max(1, (n + cols - 1) // cols)

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    axes = np.array(axes).flatten()

    for i, feat in enumerate(features):
        ax = axes[i]
        if feat not in df_b1.columns:
            ax.axis('off')
            continue
        v1 = df_b1[feat].dropna()
        v2 = df_b2[feat].dropna()
        sns.kdeplot(v1, ax=ax, color=PALETTE['B1'], fill=True,
                    alpha=0.35, linewidth=1.8, label='Bearing-1 (Real)')
        sns.kdeplot(v2, ax=ax, color=PALETTE['B2'], fill=True,
                    alpha=0.35, linewidth=1.8, label='Bearing-2 (Aug.)')
        ax.set_title(feat, fontsize=11)
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        apply_journal_style(ax)
        ax.legend(fontsize=9)

    for j in range(len(features), len(axes)):
        axes[j].axis('off')

    fig.suptitle('Feature Distribution: Bearing-1 (Real) vs Bearing-2 (Augmented)\n'
                 r'$\it{SNR_{healthy}}$=45 dB, $\it{SNR_{degrade}}$=40 dB',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    save_fig(fig, "plot0_feature_distribution.png")


# =============================================================================
# PART 2 — METRICS
# =============================================================================

def load_predictions(results_dir: str) -> dict:
    """
    Scan results_dir for preds_{Model}_{Scenario}.csv files.
    Returns dict: {(model, scenario): DataFrame}
    """
    print("\n[PART 2] Loading prediction CSV files...")
    preds = {}
    for model in MODELS:
        for scen in SCENARIOS + ["S4_SHAPGuided"]:
            fname = f"preds_{model}_{scen}.csv"
            fpath = Path(results_dir) / fname
            if fpath.exists():
                df = pd.read_csv(fpath)
                preds[(model, scen)] = df
                print(f"  [OK]      {fname}  ({len(df)} rows)")
            else:
                print(f"  [MISSING] {fname}")
    return preds


def compute_metrics(df: pd.DataFrame) -> dict:
    """
    Compute RMSE, MAE, sMAPE (%), R2, RPE_pct, Score.
    sMAPE uses a safe formula to avoid division-by-zero when BHI ≈ 0.
    """
    y_true = df["True_BHI"].values.astype(float)
    y_pred = df["Pred_BHI"].values.astype(float)

    rmse  = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae   = float(mean_absolute_error(y_true, y_pred))
    r2    = float(r2_score(y_true, y_pred))
    rpe   = float(np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + 1e-8)) * 100)

    # Safe sMAPE — prevents division by zero when both true & pred approach 0
    smape = float(np.mean(
        2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
    ) * 100)

    err   = y_pred - y_true
    score = float(np.sum(np.where(err < 0,
                                  np.exp(-err / 13) - 1,
                                  np.exp(err  / 10) - 1)))

    return {
        "RMSE"    : rmse,
        "MAE"     : mae,
        "sMAPE"   : smape,
        "R2"      : r2,
        "RPE_pct" : rpe,
        "Score"   : score,
    }


# =============================================================================
# TABLES
# =============================================================================

# ── Table 1: Comprehensive Domain Comparison (multi-level columns) ────────────

def build_table1_domain(preds: dict) -> pd.DataFrame:
    """
    Builds multi-level Table 1.
    Rows    : 4 models (LSTM, GRU, TCN, CNN-LSTM)
    Columns : Grouped by Scenario → RMSE | MAE | sMAPE | R2
    """
    print("\n[OK] Building Table 1: Comprehensive Domain Comparison...")
    metric_keys = ["RMSE", "MAE", "sMAPE", "R2"]

    rows = []
    for model in MODELS:
        row = {"Model": model}
        for scen in SCENARIOS:
            key  = (model, scen)
            slbl = SCENARIO_LABEL[scen]
            if key in preds:
                m = compute_metrics(preds[key])
                for mk in metric_keys:
                    row[f"{slbl}_{mk}"] = round(m[mk], 6)
            else:
                for mk in metric_keys:
                    row[f"{slbl}_{mk}"] = np.nan
        rows.append(row)

    df_flat = pd.DataFrame(rows).set_index("Model")

    # Build multi-level column index for readability
    tuples = []
    for scen in SCENARIOS:
        slbl = SCENARIO_LABEL[scen]
        for mk in metric_keys:
            tuples.append((slbl, mk))
    df_flat.columns = pd.MultiIndex.from_tuples(tuples)

    # Save flat version (for CSV compatibility) & multi-level display
    flat_path = os.path.join(OUTPUT_DIR, "table1_domain_comparison.csv")
    df_flat.to_csv(flat_path)
    print(f"[SAVED] table1_domain_comparison.csv")
    print(df_flat.to_string())
    return df_flat


# ── Table 2: S3 vs S4  ───────────────────────────────────────────────────────

def build_table2_s3_vs_s4(preds: dict, table1: pd.DataFrame) -> tuple:
    print("\n[OK] Building Table 2: S3 vs S4 (best model)...")

    # Find best model in S3 by RMSE
    slbl_s3 = SCENARIO_LABEL["S3_Combined"]
    rmse_col = (slbl_s3, "RMSE")
    if rmse_col in table1.columns:
        best_model = table1[rmse_col].idxmin()
    else:
        print("  [WARN] S3 RMSE column not found — defaulting to LSTM.")
        best_model = MODELS[0]

    print(f"  Best model in S3: {best_model}")

    rows = []
    for scen_key, label in [("S3_Combined", "S3 Combined"), ("S4_SHAPGuided", "S4 SHAP-Guided")]:
        key = (best_model, scen_key)
        if key in preds:
            m = compute_metrics(preds[key])
            rows.append({"Scenario": label, "RMSE": m["RMSE"], "MAE": m["MAE"],
                         "sMAPE": m["sMAPE"], "R2": m["R2"], "RPE_pct": m["RPE_pct"]})
        else:
            rows.append({"Scenario": label, "RMSE": np.nan, "MAE": np.nan,
                         "sMAPE": np.nan, "R2": np.nan, "RPE_pct": np.nan})

    df = pd.DataFrame(rows).set_index("Scenario")
    if not df["RMSE"].isna().any():
        df.loc["ΔRMSE (S4−S3)"] = df.loc["S4 SHAP-Guided"] - df.loc["S3 Combined"]

    path = os.path.join(OUTPUT_DIR, "table2_s3_vs_s4.csv")
    df.to_csv(path)
    print(f"[SAVED] table2_s3_vs_s4.csv")
    print(df.to_string())
    return df, best_model


# ── Table 3: SHAP Consistency  ───────────────────────────────────────────────

def build_table3_shap(results_dir: str) -> pd.DataFrame | None:
    print("\n[OK] Building Table 3: SHAP Cross-Scenario Consistency...")
    fpath = Path(results_dir) / "final_shap_rankings.csv"
    if not fpath.exists():
        print("  [MISSING] final_shap_rankings.csv — skipping Table 3.")
        return None

    raw = pd.read_csv(fpath)
    rank_cols = [c for c in raw.columns if c.startswith("Rank_")]
    if not rank_cols:
        print("  [WARN] No Rank_ columns found in final_shap_rankings.csv")
        return None

    df = raw[["Feature"] + rank_cols].copy()
    df["Consistency"] = df[rank_cols].std(axis=1).round(3)
    df = df.sort_values("Consistency")

    path = os.path.join(OUTPUT_DIR, "table3_shap_consistency.csv")
    df.to_csv(path, index=False)
    print(f"[SAVED] table3_shap_consistency.csv")
    print(df.head(15).to_string(index=False))
    return df


# =============================================================================
# VISUALIZATIONS
# =============================================================================

# ── Plot 1: Grouped Bar Chart (RMSE & sMAPE) ─────────────────────────────────

def plot1_grouped_bar(table1: pd.DataFrame):
    print("\n[OK] Generating Plot 1: Grouped Bar Chart (RMSE & sMAPE)...")

    # Reconstruct flat lookup from multi-level
    models = [m for m in MODELS if m in table1.index]
    if not models:
        print("  [SKIP] No model data available.")
        return

    scen_labels  = [SCENARIO_LABEL[s] for s in SCENARIOS]
    bar_colors   = ['#0072B2', '#D55E00', '#009E73']   # S1, S2, S3
    x            = np.arange(len(models))
    n_scen       = len(scen_labels)
    width        = 0.22

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for ax_idx, (metric_key, ylabel, title) in enumerate([
        ("RMSE",  "RMSE (lower is better)",  "RMSE Comparison by Domain & Architecture"),
        ("sMAPE", "sMAPE (%) (lower is better)", "sMAPE (%) Comparison by Domain & Architecture"),
    ]):
        ax = axes[ax_idx]
        for i, (slbl, color) in enumerate(zip(scen_labels, bar_colors)):
            col  = (slbl, metric_key)
            offset = (i - n_scen / 2.0 + 0.5) * width
            vals   = []
            for m in models:
                try:
                    v = table1.loc[m, col]
                    vals.append(float(v) if not pd.isna(v) else np.nan)
                except Exception:
                    vals.append(np.nan)

            valid_mask = [not np.isnan(v) for v in vals]
            x_plot     = x[valid_mask]
            v_plot     = [v for v, ok in zip(vals, valid_mask) if ok]

            bars = ax.bar(x_plot + offset, v_plot, width, label=slbl,
                          color=color, alpha=0.85, edgecolor='black', linewidth=0.7)
            for bar, val in zip(bars, v_plot):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + ax.get_ylim()[1] * 0.005,
                        f"{val:.4f}", ha='center', va='bottom', fontsize=7,
                        rotation=70)

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha='right')
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight='bold')
        ax.legend(title='Scenario', loc='upper right')
        apply_journal_style(ax)

    fig.suptitle("Model Performance Summary: RMSE & sMAPE per Scenario",
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    save_fig(fig, "plot1_rmse_smape_grouped_bar.png")


# ── Plot 2: BHI Prediction Curves — All 4 Models per Scenario (3 subplots) ───

def plot2_bhi_curves_per_scenario(preds: dict):
    """
    3 stacked subplots (S1, S2, S3). Each subplot shows:
      - True BHI (thick black)
      - Prediction from LSTM, GRU, TCN, CNN-LSTM (color + linestyle combos)
    """
    print("\n[OK] Generating Plot 2: BHI Curves per Scenario (all 4 models)...")

    fig, axes = plt.subplots(3, 1, figsize=(14, 15), sharex=False)
    fig.subplots_adjust(hspace=0.38)

    for ax_idx, scen in enumerate(SCENARIOS):
        ax    = axes[ax_idx]
        stitle = SCENARIO_TITLE[scen]
        print(f"  [OK] Generating Plot 2A for {stitle}...")

        # Ground truth: grab from first available model for this scenario
        y_true = None
        for model in MODELS:
            key = (model, scen)
            if key in preds:
                y_true = preds[key]["True_BHI"].values.astype(float)
                break

        if y_true is None:
            ax.text(0.5, 0.5, f"No data for {stitle}",
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(stitle, fontweight='bold')
            continue

        time_steps = np.arange(len(y_true))

        # Plot True BHI
        ax.plot(time_steps, y_true, color=PALETTE['true'], lw=2.8,
                label='True BHI (Actual)', zorder=10)

        # Plot each model
        legend_lines = [Line2D([0], [0], color=PALETTE['true'], lw=2.8, label='True BHI (Actual)')]
        for model in MODELS:
            key = (model, scen)
            if key not in preds:
                continue
            y_pred = preds[key]["Pred_BHI"].values.astype(float)
            m      = compute_metrics(preds[key])
            lbl    = (f"{model}  "
                      f"RMSE={m['RMSE']:.4f}  "
                      f"sMAPE={m['sMAPE']:.2f}%  "
                      f"R²={m['R2']:.4f}")
            line, = ax.plot(time_steps[:len(y_pred)], y_pred,
                            color=PALETTE[model],
                            linestyle=MODEL_LINESTYLE[model],
                            lw=1.6, alpha=0.88, label=lbl, zorder=5)
            legend_lines.append(Line2D([0], [0],
                                        color=PALETTE[model],
                                        linestyle=MODEL_LINESTYLE[model],
                                        lw=1.6, label=lbl))

        ax.set_title(stitle, fontweight='bold', fontsize=13)
        ax.set_xlabel("Time Step (Bearing-2 Test Set)", fontsize=11)
        ax.set_ylabel("Bearing Health Index (BHI)", fontsize=11)
        ax.set_ylim(-0.05, 1.12)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
        ax.legend(handles=legend_lines, loc='upper right', fontsize=8.5,
                  ncol=1, framealpha=0.95)
        apply_journal_style(ax)

        # Annotate healthy / degradation zones
        ax.axhline(y=0.8, color='#888888', linestyle=':', lw=1.0, alpha=0.6)
        ax.text(len(y_true) * 0.01, 0.81, 'Healthy Threshold (0.8)',
                fontsize=7.5, color='#555555', alpha=0.8)

    fig.suptitle("BHI Prediction Curves: All Architectures Across Feature Domain Scenarios\n"
                 "(Bearing-2 Test Set — Trained on Bearing-1)",
                 fontsize=14, fontweight='bold', y=1.01)
    save_fig(fig, "plot2_bhi_curves_per_scenario.png")


# ── Plot 3: Tail Metrics Countdown (last TAIL_N steps, S3, all 4 models) ─────

def plot3_tail_countdown(preds: dict):
    print(f"\n[OK] Generating Plot 3: Tail Metrics Countdown (last {TAIL_N} steps, S3)...")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for ax_idx, metric_key in enumerate(["RMSE", "sMAPE"]):
        ax    = axes[ax_idx]
        ylabel = "RMSE" if metric_key == "RMSE" else "sMAPE (%)"
        title  = (f"Tail RMSE — Last {TAIL_N} Steps Before Failure (S3 Combined)"
                  if metric_key == "RMSE"
                  else f"Tail sMAPE — Last {TAIL_N} Steps Before Failure (S3 Combined)")

        any_plotted = False
        for model in MODELS:
            key = (model, "S3_Combined")
            if key not in preds:
                continue
            df     = preds[key]
            y_true = df["True_BHI"].values.astype(float)
            y_pred = df["Pred_BHI"].values.astype(float)
            n_total = len(y_true)

            countdown_vals  = []
            countdown_steps = []
            for remaining in range(min(TAIL_N, n_total), 0, -1):
                start = n_total - remaining
                yt = y_true[start:]
                yp = y_pred[start:]
                if len(yt) < 3:
                    continue
                if metric_key == "RMSE":
                    val = float(np.sqrt(mean_squared_error(yt, yp)))
                else:
                    val = float(np.mean(
                        2.0 * np.abs(yt - yp) / (np.abs(yt) + np.abs(yp) + 1e-8)
                    ) * 100)
                countdown_vals.append(val)
                countdown_steps.append(remaining)

            if countdown_vals:
                ax.plot(countdown_steps, countdown_vals,
                        color=PALETTE.get(model, '#333333'),
                        linestyle=MODEL_LINESTYLE[model],
                        label=model, lw=1.8, alpha=0.88)
                any_plotted = True

        if not any_plotted:
            ax.text(0.5, 0.5, "No S3 data found", ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)

        ax.set_title(title, fontweight='bold')
        ax.set_xlabel(f"Remaining Steps Before Failure (Countdown: {TAIL_N}→0)")
        ax.set_ylabel(ylabel)
        ax.set_xlim([TAIL_N, 0])   # countdown direction
        ax.legend(title='Architecture', fontsize=9)
        apply_journal_style(ax)

    fig.suptitle("Near-Failure Prediction Accuracy: Tail Metrics Countdown\n"
                 "(S3 Combined — 4 Architectures)",
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    save_fig(fig, "plot3_tail_rmse_countdown.png")


# ── Plot 4 (NEW): Radar Chart — Holistic Multi-Metric Model Comparison ────────

def plot4_radar_metrics(preds: dict):
    """
    Radar / spider chart comparing 4 models on 4 normalised metrics in S3 Combined.
    Metrics: RMSE(inv), MAE(inv), sMAPE(inv), R2  (all normalised so bigger = better)
    """
    print("\n[OK] Generating Plot 4: Radar Chart (S3 Combined — 4 models)...")

    scen = "S3_Combined"
    metric_labels = ["RMSE↓\n(normalised)", "MAE↓\n(normalised)",
                     "sMAPE↓\n(normalised)", "R²↑\n(normalised)"]
    N = len(metric_labels)

    # Collect raw metrics
    raw = {}
    for model in MODELS:
        key = (model, scen)
        if key in preds:
            m = compute_metrics(preds[key])
            raw[model] = [m["RMSE"], m["MAE"], m["sMAPE"], m["R2"]]

    if not raw:
        print("  [SKIP] No S3 data available for radar chart.")
        return

    # Normalise: lower-is-better → invert rank, R2 stays
    arr = np.array([raw[m] for m in raw])
    norm = np.zeros_like(arr)
    for col_i in range(N):
        col = arr[:, col_i]
        if col_i < 3:   # lower = better: invert
            hi, lo = col.max(), col.min()
            rng = (hi - lo) if (hi - lo) > 1e-12 else 1e-12
            norm[:, col_i] = (hi - col) / rng
        else:           # R2: higher = better
            hi, lo = col.max(), col.min()
            rng = (hi - lo) if (hi - lo) > 1e-12 else 1e-12
            norm[:, col_i] = (col - lo) / rng

    # Radar angles
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]   # close the polygon

    fig, ax = plt.subplots(figsize=(8, 7), subplot_kw=dict(polar=True))
    ax.set_facecolor('white')

    models_in_raw = list(raw.keys())
    for idx, model in enumerate(models_in_raw):
        values = list(norm[idx]) + [norm[idx][0]]   # close
        ax.plot(angles, values, color=PALETTE[model],
                linestyle=MODEL_LINESTYLE[model], lw=2.0, label=model)
        ax.fill(angles, values, color=PALETTE[model], alpha=0.08)

    ax.set_thetagrids(np.degrees(angles[:-1]), metric_labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], fontsize=8)
    ax.tick_params(axis='y', labelcolor='#555555')
    ax.grid(True, linestyle='--', alpha=0.4, color='#999999')
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.12), fontsize=10,
              title='Architecture', frameon=True)
    ax.set_title("Holistic Model Comparison — Radar Chart\n(S3 Combined, Normalised Metrics)",
                 fontweight='bold', fontsize=12, pad=20)

    fig.tight_layout()
    save_fig(fig, "plot4_radar_metrics.png")


# ── Plot 5 (NEW): Error Distribution — Violin / Box per Model & Scenario ──────

def plot5_error_distribution(preds: dict):
    """
    Violin + box plots of absolute prediction errors across all 4 models and 3 scenarios.
    Rows = Scenarios, Columns = Models (or all-in-one faceted)
    """
    print("\n[OK] Generating Plot 5: Error Distribution (Violin/Box)...")

    records = []
    for model in MODELS:
        for scen in SCENARIOS:
            key = (model, scen)
            if key not in preds:
                continue
            df    = preds[key]
            errors = np.abs(df["True_BHI"].values - df["Pred_BHI"].values)
            for e in errors:
                records.append({
                    "Model"   : model,
                    "Scenario": SCENARIO_LABEL[scen],
                    "Abs_Error": e
                })

    if not records:
        print("  [SKIP] No prediction data available.")
        return

    err_df = pd.DataFrame(records)

    scen_labels = [SCENARIO_LABEL[s] for s in SCENARIOS]
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=False)

    for ax_idx, slbl in enumerate(scen_labels):
        ax      = axes[ax_idx]
        sub_df  = err_df[err_df["Scenario"] == slbl]
        model_order = [m for m in MODELS if m in sub_df["Model"].unique()]

        # Violin
        parts = ax.violinplot(
            [sub_df[sub_df["Model"] == m]["Abs_Error"].values for m in model_order],
            positions=range(len(model_order)),
            widths=0.55, showmedians=True, showextrema=False
        )
        for i, (pc, model) in enumerate(zip(parts['bodies'], model_order)):
            pc.set_facecolor(PALETTE[model])
            pc.set_alpha(0.55)
        parts['cmedians'].set_color('#111111')
        parts['cmedians'].set_linewidth(1.5)

        # Overlay boxplot
        bp = ax.boxplot(
            [sub_df[sub_df["Model"] == m]["Abs_Error"].values for m in model_order],
            positions=range(len(model_order)),
            widths=0.18, patch_artist=True,
            medianprops=dict(color='#111111', linewidth=2),
            whiskerprops=dict(color='#444444', linewidth=1.2),
            capprops=dict(color='#444444', linewidth=1.2),
            flierprops=dict(marker='o', markersize=2.5, alpha=0.3, color='#888888'),
            boxprops=dict(linewidth=1.2)
        )
        for patch, model in zip(bp['boxes'], model_order):
            patch.set_facecolor('white')
            patch.set_edgecolor(PALETTE[model])
            patch.set_linewidth(1.8)

        ax.set_xticks(range(len(model_order)))
        ax.set_xticklabels(model_order, rotation=15, ha='right', fontsize=10)
        ax.set_ylabel("Absolute Prediction Error" if ax_idx == 0 else "")
        ax.set_title(slbl, fontweight='bold')
        apply_journal_style(ax)

    fig.suptitle("Absolute Prediction Error Distribution per Model & Scenario\n"
                 "(Violin + Box Plot)",
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    save_fig(fig, "plot5_error_distribution.png")


# ── Plot 6 (NEW): Predicted vs Actual Scatter (S3 — all 4 models) ─────────────

def plot6_prediction_scatter(preds: dict):
    """
    2×2 scatter subplots: each panel is one model (S3 Combined).
    Diagonal = perfect prediction line. Color = density or time index.
    """
    print("\n[OK] Generating Plot 6: Prediction vs Actual Scatter (S3 Combined)...")

    scen = "S3_Combined"
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    axes_flat = axes.flatten()

    for ax_idx, model in enumerate(MODELS):
        ax  = axes_flat[ax_idx]
        key = (model, scen)

        if key not in preds:
            ax.text(0.5, 0.5, "No data", ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            ax.set_title(model, fontweight='bold')
            continue

        df     = preds[key]
        y_true = df["True_BHI"].values.astype(float)
        y_pred = df["Pred_BHI"].values.astype(float)
        m      = compute_metrics(df)

        # Colour by time index
        t_norm = np.linspace(0, 1, len(y_true))
        sc = ax.scatter(y_true, y_pred, c=t_norm, cmap='RdYlGn_r',
                        s=10, alpha=0.55, linewidths=0, zorder=4)

        # Perfect diagonal
        lims = [min(y_true.min(), y_pred.min()) - 0.02,
                max(y_true.max(), y_pred.max()) + 0.02]
        ax.plot(lims, lims, 'k--', lw=1.2, alpha=0.7, label='Perfect Prediction', zorder=5)

        # Metrics annotation box
        annot = (f"RMSE  = {m['RMSE']:.4f}\n"
                 f"MAE   = {m['MAE']:.4f}\n"
                 f"sMAPE = {m['sMAPE']:.2f}%\n"
                 f"R²    = {m['R2']:.4f}")
        ax.text(0.03, 0.97, annot, transform=ax.transAxes,
                fontsize=8.5, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.4', fc='#f7f7f7', ec='#bbbbbb', alpha=0.92))

        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("True BHI", fontsize=10)
        ax.set_ylabel("Predicted BHI", fontsize=10)
        ax.set_title(model, fontweight='bold', fontsize=12)
        apply_journal_style(ax)
        plt.colorbar(sc, ax=ax, label='Time (early→late)', pad=0.01)

    fig.suptitle("Predicted vs Actual BHI — S3 Combined (All Architectures)\n"
                 "(Color: time progression from early [green] to late [red])",
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    save_fig(fig, "plot6_prediction_scatter.png")


# ── Plot 7 (NEW): Rolling-Window RMSE Over Time ────────────────────────────────

def plot7_rolling_rmse(preds: dict, window: int = 30):
    """
    Rolling RMSE (window size = `window` steps) over the full test set for
    all 4 models and 3 scenarios (3 subplots, one per scenario).
    Reveals which architecture degrades most as bearing approaches failure.
    """
    print(f"\n[OK] Generating Plot 7: Rolling RMSE Over Time (window={window})...")

    fig, axes = plt.subplots(3, 1, figsize=(14, 13), sharex=False)
    fig.subplots_adjust(hspace=0.42)

    for ax_idx, scen in enumerate(SCENARIOS):
        ax     = axes[ax_idx]
        stitle = SCENARIO_TITLE[scen]

        any_data = False
        for model in MODELS:
            key = (model, scen)
            if key not in preds:
                continue
            df     = preds[key]
            y_true = df["True_BHI"].values.astype(float)
            y_pred = df["Pred_BHI"].values.astype(float)
            n      = len(y_true)

            # Compute rolling RMSE
            roll_rmse = []
            roll_idx  = []
            for i in range(window - 1, n):
                yt = y_true[i - window + 1: i + 1]
                yp = y_pred[i - window + 1: i + 1]
                roll_rmse.append(float(np.sqrt(mean_squared_error(yt, yp))))
                roll_idx.append(i)

            ax.plot(roll_idx, roll_rmse,
                    color=PALETTE[model],
                    linestyle=MODEL_LINESTYLE[model],
                    lw=1.6, alpha=0.88, label=model)
            any_data = True

        if not any_data:
            ax.text(0.5, 0.5, f"No data for {stitle}",
                    ha='center', va='center', transform=ax.transAxes, fontsize=11)

        ax.set_title(stitle, fontweight='bold', fontsize=12)
        ax.set_xlabel("Time Step (Bearing-2 Test Set)")
        ax.set_ylabel(f"Rolling RMSE (window={window})")
        ax.legend(title='Architecture', fontsize=9, loc='upper left')
        apply_journal_style(ax)

    fig.suptitle(f"Rolling RMSE Over Time (Window = {window} Steps)\n"
                 "Tracking Prediction Stability Across Degradation Phases",
                 fontsize=14, fontweight='bold', y=1.01)
    save_fig(fig, "plot7_rolling_rmse.png")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    # ── PART 1: Dataset Transparency ─────────────────────────────────────────
    print("\n" + "="*65)
    print("  PART 1 — DATASET TRANSPARENCY (B1 vs B2)")
    print("="*65)

    try:
        df_b1, df_b2 = load_raw_parquets()
        stats_df, avail_features = build_descriptive_stats(df_b1, df_b2)
        plot_feature_distribution(df_b1, df_b2, avail_features)
    except FileNotFoundError as e:
        print(f"  [WARN] Parquet not found — skipping Part 1. ({e})")
        df_b1, df_b2, avail_features = None, None, []

    # ── PART 2: Model Evaluation ──────────────────────────────────────────────
    print("\n" + "="*65)
    print("  PART 2 — MODEL EVALUATION & COMPARISON")
    print("="*65)

    preds = load_predictions(RESULTS_DIR)

    if not preds:
        print("\n[WARN] No prediction files found. Tables 1-3 and Plots 1-7 will be skipped.")
        print("       Run the individual modeling scripts first to generate prediction CSVs.")
        print("       Expected files: preds_{LSTM|GRU|TCN|CNNLSTM}_{S1_TimeDomain|S2_FreqDomain|S3_Combined}.csv")
    else:
        print(f"\n[INFO] Loaded {len(preds)} prediction file(s).")

        # ── Tables ────────────────────────────────────────────────────────────
        table1               = build_table1_domain(preds)
        table2, best_model   = build_table2_s3_vs_s4(preds, table1)
        table3               = build_table3_shap(RESULTS_DIR)

        # ── Plots ─────────────────────────────────────────────────────────────
        plot1_grouped_bar(table1)
        plot2_bhi_curves_per_scenario(preds)
        plot3_tail_countdown(preds)
        plot4_radar_metrics(preds)
        plot5_error_distribution(preds)
        plot6_prediction_scatter(preds)
        plot7_rolling_rmse(preds)

    print("\n" + "="*65)
    print("  [SUCCESS] 4_FINAL_SUMMARY.py completed.")
    print(f"  Output directory: {OUTPUT_DIR}")
    print("="*65)
