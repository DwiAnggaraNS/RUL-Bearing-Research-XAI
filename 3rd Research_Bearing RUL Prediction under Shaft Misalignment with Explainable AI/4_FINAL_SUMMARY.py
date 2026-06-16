"""
4_FINAL_SUMMARY.py
==================
Dataset Transparency Profiling (B1 vs B2) + Multi-Architecture RUL Model
Comparison Across Feature Domain Scenarios (S1/S2/S3/S4).

Outputs
-------
  b1_vs_b2_descriptive_stats.csv
  plot0_feature_distribution.png
  table1_domain_comparison.csv
  table2_s3_vs_s4.csv
  table3_shap_consistency.csv
  plot1_rmse_r2_grouped_bar.png
  plot2_bhi_prediction_best_model.png
  plot3_tail_rmse_countdown.png
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =============================================================================
# CONFIGURATION
# =============================================================================

TRAIN_DATA_PATH = r"D:\ProyekDosen\Riset Bearing ShaftMissalignment + XAI\bearing_1\processed_bearing1.parquet"
TEST_DATA_PATH  = r"D:\ProyekDosen\Riset Bearing ShaftMissalignment + XAI\bearing_2\processed_bearing2.parquet"
RESULTS_DIR     = r"D:\ProyekDosen\Riset Bearing ShaftMissalignment + XAI\results_LSTM"
OUTPUT_DIR      = RESULTS_DIR   # summary outputs go to same results dir

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
    'LSTM'    : '#0072B2',
    'GRU'     : '#D55E00',
    'TCN'     : '#009E73',
    'CNNLSTM' : '#CC79A7',
    'B1'      : '#0072B2',
    'B2'      : '#D55E00',
    'true'    : '#222222',
    'S1'      : '#0072B2',
    'S2'      : '#D55E00',
    'S3'      : '#009E73',
}

SCENARIO_LABEL = {
    'S1_TimeDomain' : 'S1 Time',
    'S2_FreqDomain' : 'S2 Freq',
    'S3_Combined'   : 'S3 Combined',
}


def apply_journal_style(ax):
    ax.set_facecolor('white')
    ax.minorticks_on()
    ax.grid(True, which='major', linestyle='--', linewidth=0.7, alpha=0.35, color='#b0b0b0')
    ax.grid(True, which='minor', linestyle=':',  linewidth=0.5, alpha=0.20, color='#d0d0d0')


def save_fig(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.show()
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
    print("\n[PART 1] Generating Feature Distribution Plot (KDE)...")
    n    = len(features)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    axes = axes.flatten()

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

    # hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    fig.suptitle('Feature Distribution: Bearing-1 (Real) vs Bearing-2 (Augmented)\n'
                 r'$\it{SNR_{healthy}}$=45 dB, $\it{SNR_{degrade}}$=40 dB',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    save_fig(fig, "plot0_feature_distribution.png")


# =============================================================================
# PART 2 — MODEL EVALUATION & COMPARISON
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
    y_true = df["True_BHI"].values
    y_pred = df["Pred_BHI"].values
    rmse   = np.sqrt(mean_squared_error(y_true, y_pred))
    mae    = mean_absolute_error(y_true, y_pred)
    r2     = r2_score(y_true, y_pred)
    rpe    = np.mean(np.abs(y_true - y_pred) / (y_true + 1e-8)) * 100
    err    = y_pred - y_true
    score  = float(np.sum(np.where(err < 0, np.exp(-err / 13) - 1, np.exp(err / 10) - 1)))
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "RPE_pct": rpe, "Score": score}


# ── Table 1: Domain Comparison  ───────────────────────────────────────────────

def build_table1_domain(preds: dict) -> pd.DataFrame:
    print("\n[PART 2] Building Table 1: Domain Comparison...")
    rows = []
    for model in MODELS:
        row = {"Model": model}
        for scen in SCENARIOS:
            key  = (model, scen)
            slbl = SCENARIO_LABEL[scen]
            if key in preds:
                m = compute_metrics(preds[key])
                row[f"RMSE_{slbl}"] = round(m["RMSE"], 6)
                row[f"R2_{slbl}"]   = round(m["R2"],   6)
                row[f"MAE_{slbl}"]  = round(m["MAE"],  6)
            else:
                row[f"RMSE_{slbl}"] = np.nan
                row[f"R2_{slbl}"]   = np.nan
                row[f"MAE_{slbl}"]  = np.nan
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Model")
    path = os.path.join(OUTPUT_DIR, "table1_domain_comparison.csv")
    df.to_csv(path)
    print(f"[SAVED] table1_domain_comparison.csv")
    print(df.to_string())
    return df


# ── Table 2: S3 vs S4  ───────────────────────────────────────────────────────

def build_table2_s3_vs_s4(preds: dict, table1: pd.DataFrame) -> pd.DataFrame:
    print("\n[PART 2] Building Table 2: S3 vs S4 (best model)...")

    slbl_s3 = SCENARIO_LABEL["S3_Combined"]
    rmse_col = f"RMSE_{slbl_s3}"

    if rmse_col in table1.columns:
        best_model = table1[rmse_col].idxmin()
    else:
        print("  [WARN] S3 RMSE column not found — defaulting to first model.")
        best_model = MODELS[0]

    print(f"  Best model in S3: {best_model}")

    rows = []
    for scen_key, label in [("S3_Combined", "S3 Combined"), ("S4_SHAPGuided", "S4 SHAP-Guided")]:
        key = (best_model, scen_key)
        if key in preds:
            m = compute_metrics(preds[key])
            rows.append({"Scenario": label, "RMSE": m["RMSE"], "MAE": m["MAE"],
                         "R2": m["R2"], "RPE_pct": m["RPE_pct"]})
        else:
            rows.append({"Scenario": label, "RMSE": np.nan, "MAE": np.nan,
                         "R2": np.nan, "RPE_pct": np.nan})

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
    print("\n[PART 2] Building Table 3: SHAP Cross-Scenario Consistency...")
    fpath = Path(results_dir) / "final_shap_rankings.csv"
    if not fpath.exists():
        print("  [MISSING] final_shap_rankings.csv — skipping Table 3.")
        return None

    raw = pd.read_csv(fpath)

    # Expected columns: Feature, Rank_S1 Time Domain, Importance_S1 Time Domain, ...
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
# PART 2 — VISUALIZATIONS
# =============================================================================

# ── Plot 1: Grouped Bar Chart (RMSE & R²)  ───────────────────────────────────

def plot1_grouped_bar(table1: pd.DataFrame):
    print("\n[PART 2] Generating Plot 1: Grouped Bar Chart (RMSE & R²)...")
    models = [m for m in MODELS if m in table1.index]
    if not models:
        print("  [SKIP] No model data available.")
        return

    scen_labels  = [SCENARIO_LABEL[s] for s in SCENARIOS]
    rmse_cols    = [f"RMSE_{l}" for l in scen_labels]
    r2_cols      = [f"R2_{l}"   for l in scen_labels]
    bar_colors   = ['#0072B2', '#D55E00', '#009E73']   # S1, S2, S3
    x            = np.arange(len(models))
    n_scen       = len(scen_labels)
    width        = 0.22

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax_idx, (metric_cols, ylabel, title, ascending) in enumerate([
        (rmse_cols, "RMSE",    "RMSE Comparison by Domain (Lower is Better)",  True),
        (r2_cols,   "R² Score","R² Score Comparison by Domain (Higher is Better)", False),
    ]):
        ax = axes[ax_idx]
        for i, (col, label, color) in enumerate(zip(metric_cols, scen_labels, bar_colors)):
            offset = (i - n_scen // 2) * width + (width / 2 if n_scen % 2 == 0 else 0)
            vals   = [table1.loc[m, col] if (m in table1.index and col in table1.columns)
                      else np.nan for m in models]
            bars   = ax.bar(x + offset, vals, width, label=label,
                            color=color, alpha=0.85, edgecolor='black', linewidth=0.7)
            for bar, val in zip(bars, vals):
                if not np.isnan(val):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.002,
                            f"{val:.4f}", ha='center', va='bottom', fontsize=7.5)

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha='right')
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight='bold')
        ax.legend(title='Scenario', loc='upper right')
        apply_journal_style(ax)

    fig.tight_layout()
    save_fig(fig, "plot1_rmse_r2_grouped_bar.png")


# ── Plot 2: BHI Prediction Curves (best model, S1/S2/S3 overlay)  ────────────

def plot2_bhi_curves(preds: dict, best_model: str):
    print(f"\n[PART 2] Generating Plot 2: BHI Curves for best model ({best_model})...")

    avail_scens = [s for s in SCENARIOS if (best_model, s) in preds]
    if not avail_scens:
        print("  [SKIP] No prediction data for best model.")
        return

    # Use ground truth from first available scenario
    ref_df = preds[(best_model, avail_scens[0])]
    y_true = ref_df["True_BHI"].values

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(y_true, color=PALETTE['true'], lw=2.2, label='True BHI (Actual)', zorder=5)

    line_styles = ['-', '--', ':']
    scen_colors = [PALETTE['S1'], PALETTE['S2'], PALETTE['S3']]

    for i, scen in enumerate(avail_scens):
        df    = preds[(best_model, scen)]
        m     = compute_metrics(df)
        lbl   = f"{SCENARIO_LABEL[scen]}  RMSE={m['RMSE']:.4f}  R²={m['R2']:.4f}"
        ax.plot(df["Pred_BHI"].values, color=scen_colors[i], lw=1.7,
                linestyle=line_styles[i], alpha=0.88, label=lbl, zorder=4)

    ax.set_title(f"BHI Prediction — {best_model} (Best Model in S3) on Bearing-2",
                 fontweight='bold')
    ax.set_xlabel("Time Step (Bearing-2 Test Set)")
    ax.set_ylabel("Bearing Health Index (BHI)")
    ax.set_ylim(-0.05, 1.12)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.legend(loc='upper right', fontsize=9)
    apply_journal_style(ax)
    fig.tight_layout()
    save_fig(fig, "plot2_bhi_prediction_best_model.png")


# ── Plot 3: Tail RMSE Countdown (last TAIL_N steps)  ─────────────────────────

def plot3_tail_countdown(preds: dict):
    print(f"\n[PART 2] Generating Plot 3: Tail RMSE Countdown (last {TAIL_N} steps)...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax_idx, metric_key in enumerate(["RMSE", "R2"]):
        ax     = axes[ax_idx]
        ylabel = "RMSE" if metric_key == "RMSE" else "R² Score"
        title  = (f"Tail RMSE (Last {TAIL_N} Steps) — Countdown to Failure"
                  if metric_key == "RMSE"
                  else f"Tail R² (Last {TAIL_N} Steps) — Countdown to Failure")

        for model in MODELS:
            key = (model, "S3_Combined")
            if key not in preds:
                continue
            df     = preds[key]
            y_true = df["True_BHI"].values
            y_pred = df["Pred_BHI"].values
            n_total = len(y_true)

            # Compute rolling metric from tail index i to end
            countdown_vals  = []
            countdown_steps = []
            for remaining in range(min(TAIL_N, n_total), 0, -1):
                start = n_total - remaining
                yt = y_true[start:]
                yp = y_pred[start:]
                if len(yt) < 3:
                    continue
                if metric_key == "RMSE":
                    val = np.sqrt(mean_squared_error(yt, yp))
                else:
                    val = r2_score(yt, yp)
                countdown_vals.append(val)
                countdown_steps.append(remaining)

            ax.plot(countdown_steps, countdown_vals,
                    color=PALETTE.get(model, '#333333'),
                    label=model, lw=1.7, alpha=0.88)

        ax.set_title(title, fontweight='bold')
        ax.set_xlabel(f"Remaining Steps Before Failure (Countdown: {TAIL_N}→0)")
        ax.set_ylabel(ylabel)
        ax.set_xlim([TAIL_N, 0])   # X-axis: countdown direction
        ax.legend(title='Architecture', fontsize=9)
        apply_journal_style(ax)

    fig.tight_layout()
    save_fig(fig, "plot3_tail_rmse_countdown.png")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    # ── PART 1: Dataset Transparency ─────────────────────────────────────────
    print("\n" + "="*65)
    print("  PART 1 — DATASET TRANSPARENCY (B1 vs B2)")
    print("="*65)

    df_b1, df_b2 = load_raw_parquets()
    stats_df, avail_features = build_descriptive_stats(df_b1, df_b2)
    plot_feature_distribution(df_b1, df_b2, avail_features)

    # ── PART 2: Model Evaluation ──────────────────────────────────────────────
    print("\n" + "="*65)
    print("  PART 2 — MODEL EVALUATION & COMPARISON")
    print("="*65)

    preds = load_predictions(RESULTS_DIR)

    if not preds:
        print("\n[WARN] No prediction files found. Tables 1-3 and Plots 1-3 will be skipped.")
        print("       Run the individual modeling scripts first to generate prediction CSVs.")
    else:
        table1                  = build_table1_domain(preds)
        table2, best_model      = build_table2_s3_vs_s4(preds, table1)
        table3                  = build_table3_shap(RESULTS_DIR)

        plot1_grouped_bar(table1)
        plot2_bhi_curves(preds, best_model)
        plot3_tail_countdown(preds)

    print("\n" + "="*65)
    print("  [SUCCESS] 4_FINAL_SUMMARY.py completed.")
    print("="*65)
