import os

os.environ['KMP_DUPLICATE_LIB_OK']    = 'True'
os.environ["OMP_NUM_THREADS"]          = "1"
os.environ["CUDA_LAUNCH_BLOCKING"]     = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]  = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import shap
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

torch.backends.cudnn.enabled       = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False


# =============================================================================
# 1. GLOBAL CONFIGURATION
# =============================================================================

class Config:
    TRAIN_DATA_PATH = r"D:\ProyekDosen\Riset Bearing ShaftMissalignment + XAI\bearing_1\processed_bearing1.parquet"
    TEST_DATA_PATH  = r"D:\ProyekDosen\Riset Bearing ShaftMissalignment + XAI\bearing_2\processed_bearing2.parquet"
    OUTPUT_DIR      = r"D:\ProyekDosen\Riset Bearing ShaftMissalignment + XAI\results_LSTM"

    TARGET_COL      = 'bhi'
    META_COLS       = ['segment', 'time_s', 'time_min', 'bhi', 'label_vcd',
                       'T_cp', 'T_f', 'rms_ema_x', 'rms_ema_y', 'rms_ema_z']

    # Fixed hyperparameters (literature-based)
    WINDOW_SIZE     = 15
    BATCH_SIZE      = 32
    HIDDEN_SIZE     = 64
    NUM_LAYERS      = 2
    DROPOUT         = 0.0
    LEARNING_RATE   = 0.001
    EPOCHS          = 30

    TOP_N_FEATURES  = 10
    BEARING_LIFE_S  = 392275

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
print(f"[INFO] Device  : {Config.DEVICE}")
print(f"[INFO] Output  : {Config.OUTPUT_DIR}")


# =============================================================================
# 2. FEATURE DATA LOADER (OOP)
# =============================================================================

class BearingDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, window_size: int):
        self.X, self.y = X, y
        self.window_size = window_size
        self.indices = list(range(0, len(X) - window_size))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        s = self.indices[idx]
        e = s + self.window_size
        return (
            torch.tensor(self.X[s:e], dtype=torch.float32),
            torch.tensor([self.y[e - 1]], dtype=torch.float32),
        )


class FeatureDataLoader:
    """
    Loads B1 & B2 parquet files, fits MinMaxScaler on B1,
    and exposes methods to build domain-filtered DataLoaders.
    Prefix convention: td_ = time-domain, fd_ = frequency-domain.
    """

    def __init__(self):
        print("[INFO] Loading datasets...")
        df_b1 = pd.read_parquet(Config.TRAIN_DATA_PATH)
        df_b2 = pd.read_parquet(Config.TEST_DATA_PATH)

        self.y_b1 = df_b1[Config.TARGET_COL].values
        self.y_b2 = df_b2[Config.TARGET_COL].values

        feature_pool = [c for c in df_b1.columns if c not in Config.META_COLS]
        self.all_features = feature_pool

        scaler = MinMaxScaler()
        X_b1_raw = df_b1[feature_pool].values
        X_b2_raw = df_b2[feature_pool].values
        self._X_b1 = scaler.fit_transform(X_b1_raw)
        self._X_b2 = scaler.transform(X_b2_raw)

        self.td_cols   = [c for c in feature_pool if c.startswith("td_")]
        self.fd_cols   = [c for c in feature_pool if c.startswith("fd_")]
        self.all_col_map = {c: i for i, c in enumerate(feature_pool)}

        print(f"[INFO] Total features : {len(feature_pool)}")
        print(f"[INFO] Time-domain    : {len(self.td_cols)} features")
        print(f"[INFO] Freq-domain    : {len(self.fd_cols)} features")

    def _select_cols(self, cols: list):
        idx = [self.all_col_map[c] for c in cols]
        return self._X_b1[:, idx], self._X_b2[:, idx]

    def _make_loaders(self, X_b1, X_b2, y_b1, y_b2):
        ds_train = BearingDataset(X_b1, y_b1, Config.WINDOW_SIZE)
        ds_test  = BearingDataset(X_b2, y_b2, Config.WINDOW_SIZE)
        dl_train = DataLoader(ds_train, batch_size=Config.BATCH_SIZE, shuffle=True)
        dl_test  = DataLoader(ds_test,  batch_size=Config.BATCH_SIZE, shuffle=False)
        return dl_train, ds_train, dl_test, ds_test

    def get_time_domain_loaders(self):
        X_b1, X_b2 = self._select_cols(self.td_cols)
        return self._make_loaders(X_b1, X_b2, self.y_b1, self.y_b2), self.td_cols

    def get_freq_domain_loaders(self):
        X_b1, X_b2 = self._select_cols(self.fd_cols)
        return self._make_loaders(X_b1, X_b2, self.y_b1, self.y_b2), self.fd_cols

    def get_combined_loaders(self):
        combined = self.td_cols + self.fd_cols
        X_b1, X_b2 = self._select_cols(combined)
        return self._make_loaders(X_b1, X_b2, self.y_b1, self.y_b2), combined

    def get_custom_feature_loaders(self, feature_list: list):
        X_b1, X_b2 = self._select_cols(feature_list)
        return self._make_loaders(X_b1, X_b2, self.y_b1, self.y_b2), feature_list


# =============================================================================
# 3. MODEL ARCHITECTURE  (modular — swap recurrent block for GRU/TCN/CNN-LSTM)
# =============================================================================

class RULLSTM(nn.Module):
    """Modular LSTM for RUL/BHI regression. Replace self.recurrent to adapt architecture."""

    def __init__(self, input_size: int, hidden_size: int = Config.HIDDEN_SIZE,
                 num_layers: int = Config.NUM_LAYERS, dropout: float = Config.DROPOUT):
        super().__init__()
        self.recurrent = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.recurrent(x)
        return self.head(out[:, -1, :])


def build_model(input_size: int) -> nn.Module:
    return RULLSTM(
        input_size  = input_size,
        hidden_size = Config.HIDDEN_SIZE,
        num_layers  = Config.NUM_LAYERS,
        dropout     = Config.DROPOUT,
    ).to(Config.DEVICE)


# =============================================================================
# 4. TRAINING & EVALUATION
# =============================================================================

def train_one_epoch(model, loader, optimizer, criterion) -> float:
    model.train()
    losses = []
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(Config.DEVICE), y_b.to(Config.DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(X_b), y_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def evaluate_loader(model, loader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            preds.extend(model(X_b.to(Config.DEVICE)).cpu().numpy().flatten())
            targets.extend(y_b.numpy().flatten())
    return np.array(targets), np.array(preds)


def train_model(model, train_loader, verbose: bool = True) -> list:
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=7, min_lr=1e-5
    )
    history = []
    for epoch in range(Config.EPOCHS):
        avg_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        history.append(avg_loss)
        scheduler.step(avg_loss)
        if verbose and ((epoch + 1) % 5 == 0 or epoch == 0):
            print(f"  Epoch [{epoch+1:>3}/{Config.EPOCHS}] Loss: {avg_loss:.6f}"
                  f" | LR: {optimizer.param_groups[0]['lr']:.2e}")
    return history


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    mae   = mean_absolute_error(y_true, y_pred)
    r2    = r2_score(y_true, y_pred)
    rpe   = np.mean(np.abs(y_true - y_pred) / (y_true + 1e-8)) * 100
    err   = y_pred - y_true
    score = float(np.sum(np.where(err < 0, np.exp(-err / 13) - 1, np.exp(err / 10) - 1)))
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "RPE_pct": rpe, "Score": score}


def run_scenario(label: str, dl_train, ds_train, dl_test, ds_test,
                 feature_names: list, pred_filename: str,
                 shap_rankings: dict) -> dict:
    """
    Full pipeline for one scenario:
    train on B1 → evaluate on B2 → save predictions CSV → run SHAP.
    Returns metrics dict.
    """
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    n_features = len(feature_names)
    model = build_model(n_features)
    print(f"[INFO] Features: {n_features} | Training on B1...")

    loss_hist = train_model(model, dl_train)
    plot_loss_history(loss_hist, tag=label)

    y_true, y_pred = evaluate_loader(model, dl_test)
    metrics = calculate_metrics(y_true, y_pred)

    print(f"\n[METRICS] {label}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save predictions CSV
    df_pred = pd.DataFrame({
        "True_BHI"         : y_true,
        "Pred_BHI"         : y_pred,
        "True_RUL_Seconds" : y_true * Config.BEARING_LIFE_S,
        "Pred_RUL_Seconds" : y_pred * Config.BEARING_LIFE_S,
    })
    pred_path = os.path.join(Config.OUTPUT_DIR, pred_filename)
    df_pred.to_csv(pred_path, index=False)
    print(f"[INFO] Predictions saved: {pred_path}")

    plot_bhi_prediction(y_true, y_pred, metrics, tag=label)

    # SHAP
    importance = run_shap(model, ds_test, feature_names, tag=label)
    shap_rankings[label] = importance  # ranked pd.Series

    del model
    return metrics


# =============================================================================
# 5. VISUALIZATIONS — Publication Style
# =============================================================================

PAPER_RC = {
    'font.family'      : 'DejaVu Sans',
    'font.size'        : 11,
    'axes.titlesize'   : 12,
    'axes.labelsize'   : 11,
    'xtick.labelsize'  : 10,
    'ytick.labelsize'  : 10,
    'legend.fontsize'  : 10,
    'figure.dpi'       : 150,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'axes.grid'        : True,
    'grid.alpha'       : 0.3,
    'grid.linestyle'   : '--',
    'lines.linewidth'  : 1.8,
}

PALETTE = {
    'true'      : '#2C3E50',
    'pred'      : '#E74C3C',
    'shap'      : '#117A65',
    'error_fill': '#E74C3C',
}


def plot_bhi_prediction(y_true: np.ndarray, y_pred: np.ndarray,
                        metrics: dict, tag: str = ""):
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)

    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.plot(y_true, color=PALETTE['true'], lw=2.0, label='True BHI', zorder=3)
        ax.plot(y_pred, color=PALETTE['pred'], lw=1.6, linestyle='--',
                alpha=0.9, label='Predicted BHI', zorder=2)
        ax.fill_between(np.arange(len(y_true)), y_true, y_pred,
                        alpha=0.12, color=PALETTE['error_fill'], label='Error')

        metric_txt = (f"RMSE={metrics['RMSE']:.4f}  "
                      f"MAE={metrics['MAE']:.4f}  "
                      f"R²={metrics['R2']:.4f}")
        ax.set_title(f'BHI Prediction — {tag}\n{metric_txt}', fontweight='bold', pad=10)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Health Index (BHI)')
        ax.set_ylim(-0.05, 1.12)
        ax.legend(loc='upper right', framealpha=0.9)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
        plt.tight_layout()

        out = os.path.join(Config.OUTPUT_DIR, f'plot_bhi_{tag.replace(" ", "_")}.png')
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.show()
        print(f'[INFO] BHI plot saved: {out}')


def plot_loss_history(loss_hist: list, tag: str = ""):
    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(loss_hist, color='#2471A3')
        ax.set_title(f'Training Loss — {tag}', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')
        plt.tight_layout()
        out = os.path.join(Config.OUTPUT_DIR, f'loss_{tag.replace(" ", "_")}.png')
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.show()


# =============================================================================
# 6. XAI — SHAP (GradientExplainer + Beeswarm + Bar)
# =============================================================================

def _get_xai_tensors(dataset: BearingDataset, n_bg: int = 200, n_test: int = 50):
    n_avail = len(dataset)
    n_bg    = min(n_bg,   n_avail // 2)
    n_test  = min(n_test, n_avail - n_bg)
    bg   = torch.stack([dataset[i][0] for i in range(n_bg)]).to(Config.DEVICE)
    test = torch.stack([dataset[i][0] for i in range(n_bg, n_bg + n_test)]).to(Config.DEVICE)
    return bg, test


def run_shap(model: nn.Module, dataset: BearingDataset,
             feature_names: list, tag: str = "",
             top_n: int = Config.TOP_N_FEATURES) -> pd.Series:
    """GradientExplainer SHAP: plots beeswarm & bar chart, returns ranked importance Series."""
    print("\n[INFO] Running SHAP GradientExplainer...")
    model.eval()
    bg_tensor, test_tensor = _get_xai_tensors(dataset)

    explainer = shap.GradientExplainer(model, bg_tensor)
    shap_vals = explainer.shap_values(test_tensor)

    sv = shap_vals[0] if isinstance(shap_vals, list) else shap_vals
    if sv.ndim == 4:
        sv = sv.squeeze(axis=-1)   # (N, W, F)

    sv_flat = sv.mean(axis=1)      # (N, F)

    # ── Beeswarm ──────────────────────────────────────────────────────────────
    expl_obj = shap.Explanation(
        values       = sv_flat,
        base_values  = np.zeros(sv_flat.shape[0]),
        data         = test_tensor.cpu().numpy().mean(axis=1),
        feature_names= feature_names,
    )
    safe_tag = tag.replace(" ", "_")
    with plt.rc_context(PAPER_RC):
        plt.figure(figsize=(10, 8))
        shap.plots.beeswarm(expl_obj, max_display=top_n, show=False)
        plt.title(f"SHAP Beeswarm — {tag}", fontweight='bold', pad=10)
        plt.tight_layout()
        out_bee = os.path.join(Config.OUTPUT_DIR, f'shap_beeswarm_{safe_tag}.png')
        plt.savefig(out_bee, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"[INFO] SHAP beeswarm saved: {out_bee}")

    # ── Bar importance ────────────────────────────────────────────────────────
    importance = pd.Series(
        np.abs(sv).mean(axis=(0, 1)), index=feature_names
    ).sort_values(ascending=False)

    top_imp = importance.head(top_n).sort_values(ascending=True)
    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(8, max(5, top_n * 0.4)))
        bars = ax.barh(top_imp.index, top_imp.values,
                       color=PALETTE['shap'], edgecolor='none', height=0.65, alpha=0.88)
        for bar, val in zip(bars, top_imp.values):
            ax.text(val + top_imp.values.max() * 0.015,
                    bar.get_y() + bar.get_height() / 2,
                    f'{val:.4f}', va='center', fontsize=8.5)
        ax.set_title(f'SHAP Feature Importance — Top {top_n} | {tag}', fontweight='bold')
        ax.set_xlabel('Mean |SHAP Value|')
        plt.tight_layout()
        out_bar = os.path.join(Config.OUTPUT_DIR, f'shap_bar_{safe_tag}.png')
        plt.savefig(out_bar, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"[INFO] SHAP bar saved: {out_bar}")

    print(f"\n[INFO] Top-{top_n} SHAP features ({tag}):")
    for f, v in importance.head(top_n).items():
        print(f"  {f:<40s} {v:.6f}")

    return importance


# =============================================================================
# 7. OUTPUT AGGREGATION
# =============================================================================

def save_metrics_table(results: dict):
    rows = []
    for scenario, metrics in results.items():
        row = {"Scenario": scenario}
        row.update({k: round(v, 6) for k, v in metrics.items()})
        rows.append(row)
    df = pd.DataFrame(rows).set_index("Scenario")

    print("\n" + "="*70)
    print("FINAL METRICS COMPARISON")
    print("="*70)
    print(df.to_string())

    path = os.path.join(Config.OUTPUT_DIR, "final_metrics_comparison.csv")
    df.to_csv(path)
    print(f"\n[INFO] Metrics table saved: {path}")
    return df


def save_shap_rankings(shap_rankings: dict):
    """
    Merges per-scenario SHAP importance into one cross-scenario ranking table.
    Columns: Feature | Rank_S1 | Importance_S1 | Rank_S2 | Importance_S2 | ...
    """
    frames = []
    for scenario, importance in shap_rankings.items():
        rank_df = importance.reset_index()
        rank_df.columns = ["Feature", f"Importance_{scenario}"]
        rank_df[f"Rank_{scenario}"] = range(1, len(rank_df) + 1)
        rank_df = rank_df.set_index("Feature")
        frames.append(rank_df)

    combined = pd.concat(frames, axis=1).reset_index()

    # Reorder columns: Feature, then alternating Rank/Importance per scenario
    ordered_cols = ["Feature"]
    for scenario in shap_rankings.keys():
        ordered_cols += [f"Rank_{scenario}", f"Importance_{scenario}"]
    combined = combined[ordered_cols]

    path = os.path.join(Config.OUTPUT_DIR, "final_shap_rankings.csv")
    combined.to_csv(path, index=False)
    print(f"[INFO] Cross-scenario SHAP rankings saved: {path}")
    return combined


# =============================================================================
# MAIN EXECUTION — Feature Domain Experiment (S1–S3) + Custom (S4 placeholder)
# =============================================================================

if __name__ == "__main__":

    # ── Data Loading ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  DATA LOADING")
    print("="*60)

    fdl = FeatureDataLoader()

    scenario_results = {}
    shap_rankings    = {}

    # ── Scenario 1: Time Domain Baseline ─────────────────────────────────────
    (dl_train, ds_train, dl_test, ds_test), feat_cols = fdl.get_time_domain_loaders()
    metrics_s1 = run_scenario(
        label        = "S1 Time Domain",
        dl_train     = dl_train,
        ds_train     = ds_train,
        dl_test      = dl_test,
        ds_test      = ds_test,
        feature_names= feat_cols,
        pred_filename = "preds_S1_TimeDomain.csv",
        shap_rankings= shap_rankings,
    )
    scenario_results["S1_TimeDomain"] = metrics_s1

    # ── Scenario 2: Frequency Domain Baseline ────────────────────────────────
    (dl_train, ds_train, dl_test, ds_test), feat_cols = fdl.get_freq_domain_loaders()
    metrics_s2 = run_scenario(
        label        = "S2 Freq Domain",
        dl_train     = dl_train,
        ds_train     = ds_train,
        dl_test      = dl_test,
        ds_test      = ds_test,
        feature_names= feat_cols,
        pred_filename = "preds_S2_FreqDomain.csv",
        shap_rankings= shap_rankings,
    )
    scenario_results["S2_FreqDomain"] = metrics_s2

    # ── Scenario 3: Combined Baseline ────────────────────────────────────────
    (dl_train, ds_train, dl_test, ds_test), feat_cols = fdl.get_combined_loaders()
    metrics_s3 = run_scenario(
        label        = "S3 Combined",
        dl_train     = dl_train,
        ds_train     = ds_train,
        dl_test      = dl_test,
        ds_test      = ds_test,
        feature_names= feat_cols,
        pred_filename = "preds_S3_Combined.csv",
        shap_rankings= shap_rankings,
    )
    scenario_results["S3_Combined"] = metrics_s3

    # ── Scenario 4: Custom / SHAP-Guided (placeholder — uncomment to activate) ─
    # top_features = list(shap_rankings["S3 Combined"].head(Config.TOP_N_FEATURES).index)
    # (dl_train, ds_train, dl_test, ds_test), feat_cols = fdl.get_custom_feature_loaders(top_features)
    # metrics_s4 = run_scenario(
    #     label        = "S4 SHAP-Guided",
    #     dl_train     = dl_train,
    #     ds_train     = ds_train,
    #     dl_test      = dl_test,
    #     ds_test      = ds_test,
    #     feature_names= feat_cols,
    #     pred_filename = "preds_S4_SHAPGuided.csv",
    #     shap_rankings= shap_rankings,
    # )
    # scenario_results["S4_SHAPGuided"] = metrics_s4

    # ── Final Outputs ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  FINAL OUTPUTS")
    print("="*60)

    save_metrics_table(scenario_results)
    save_shap_rankings(shap_rankings)

    print("\n[SUCCESS] All scenarios completed successfully!")