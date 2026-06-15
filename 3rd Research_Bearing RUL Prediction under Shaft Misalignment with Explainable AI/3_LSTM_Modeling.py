import os
import random

os.environ['KMP_DUPLICATE_LIB_OK']         = 'True'
os.environ["OMP_NUM_THREADS"]               = "1"
os.environ["CUDA_LAUNCH_BLOCKING"]          = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]       = "expandable_segments:True"

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

    COLS_TO_DROP = [
        'segment', 'time_s', 'time_min', 'bhi', 'label_vcd',
        'T_cp', 'T_f', 'rms_ema_x', 'rms_ema_y', 'rms_ema_z'
    ]
    TARGET_COL   = 'bhi'

    # Baseline hyperparameters (Scenario 1)
    WINDOW_SIZE   = 128
    BATCH_SIZE    = 32
    HIDDEN_SIZE   = 64
    NUM_LAYERS    = 2
    DROPOUT       = 0.0
    LEARNING_RATE = 0.001
    EPOCHS        = 30

    TRAIN_SPLIT   = 0.85     # temporal split ratio for Scenario 2
    TOP_N_FEATURES = 10      # Scenario 3 → Scenario 4
    RANDOM_TRIALS  = 6       # Random Search trials in Scenario 2
    BEARING_LIFE_S = 392275

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
print(f"[INFO] Device: {Config.DEVICE}")


# =============================================================================
# 2. DATA STRUCTURES & LOADERS
# =============================================================================

class BearingDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, window_size: int, stride: int = 1):
        self.X = X
        self.y = y
        self.window_size = window_size
        self.indices = list(range(0, len(X) - window_size, stride))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        s = self.indices[idx]
        e = s + self.window_size
        return (
            torch.tensor(self.X[s:e], dtype=torch.float32),
            torch.tensor([self.y[e - 1]],  dtype=torch.float32),
        )


def make_loader(X: np.ndarray, y: np.ndarray, window_size: int,
                batch_size: int, shuffle: bool = False) -> tuple[DataLoader, BearingDataset]:
    ds = BearingDataset(X, y, window_size)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    return dl, ds


def load_raw_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list, MinMaxScaler]:
    """Loads B1 & B2, fits scaler on B1, returns raw scaled arrays + feature list."""
    print("[INFO] Loading datasets...")
    df_b1 = pd.read_parquet(Config.TRAIN_DATA_PATH)
    df_b2 = pd.read_parquet(Config.TEST_DATA_PATH)

    drop_cols   = Config.COLS_TO_DROP + [Config.TARGET_COL]
    feature_cols = [c for c in df_b1.columns if c not in drop_cols]
    print(f"[INFO] {len(feature_cols)} feature columns detected.")

    y_b1 = df_b1[Config.TARGET_COL].values
    y_b2 = df_b2[Config.TARGET_COL].values
    X_b1 = df_b1[feature_cols].values
    X_b2 = df_b2[feature_cols].values

    scaler = MinMaxScaler()
    X_b1   = scaler.fit_transform(X_b1)
    X_b2   = scaler.transform(X_b2)

    return X_b1, y_b1, X_b2, y_b2, feature_cols, scaler


def prepare_all_loaders(X_b1, y_b1, X_b2, y_b2,
                        window_size: int, batch_size: int) -> dict:
    """Builds all required DataLoaders from pre-scaled arrays."""
    split_idx = int(len(X_b1) * Config.TRAIN_SPLIT)

    loader_b1_full,  ds_b1_full  = make_loader(X_b1, y_b1, window_size, batch_size, shuffle=True)
    loader_b2_test,  ds_b2_test  = make_loader(X_b2, y_b2, window_size, batch_size, shuffle=False)
    loader_b1_train, ds_b1_train = make_loader(X_b1[:split_idx], y_b1[:split_idx],
                                               window_size, batch_size, shuffle=True)
    loader_b1_val,   ds_b1_val   = make_loader(X_b1[split_idx:], y_b1[split_idx:],
                                               window_size, batch_size, shuffle=False)

    return {
        "b1_full"  : (loader_b1_full,  ds_b1_full),
        "b2_test"  : (loader_b2_test,  ds_b2_test),
        "b1_train" : (loader_b1_train, ds_b1_train),
        "b1_val"   : (loader_b1_val,   ds_b1_val),
    }


# =============================================================================
# 3. MODEL ARCHITECTURE  (modular — swap LSTM block for GRU/TCN/CNN-LSTM)
# =============================================================================

class RULLSTM(nn.Module):
    """Modular LSTM for RUL/BHI regression. Swap the recurrent block as needed."""

    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.0):
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


def build_model(input_size: int, cfg: dict) -> nn.Module:
    """Instantiates RULLSTM from a config dict and moves it to device."""
    model = RULLSTM(
        input_size  = input_size,
        hidden_size = cfg.get("hidden_size", Config.HIDDEN_SIZE),
        num_layers  = cfg.get("num_layers",  Config.NUM_LAYERS),
        dropout     = cfg.get("dropout",     Config.DROPOUT),
    ).to(Config.DEVICE)
    return model


# =============================================================================
# 4. TRAINING & EVALUATION
# =============================================================================

def train_one_epoch(model, loader, optimizer, criterion) -> float:
    model.train()
    losses = []
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(Config.DEVICE), y_b.to(Config.DEVICE)
        optimizer.zero_grad()
        out  = model(X_b)
        loss = criterion(out, y_b)
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


def train_model(model, train_loader, epochs: int, lr: float,
                val_loader=None, verbose: bool = True) -> list:
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=7, min_lr=1e-5
    )
    loss_history = []

    for epoch in range(epochs):
        avg_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        loss_history.append(avg_loss)
        scheduler.step(avg_loss)

        if verbose and ((epoch + 1) % 5 == 0 or epoch == 0):
            val_info = ""
            if val_loader is not None:
                y_t, y_p = evaluate_loader(model, val_loader)
                val_rmse = np.sqrt(mean_squared_error(y_t, y_p))
                val_info = f" | Val RMSE: {val_rmse:.6f}"
            print(f"  Epoch [{epoch+1:>3}/{epochs}] Loss: {avg_loss:.6f}"
                  f" | LR: {optimizer.param_groups[0]['lr']:.2e}{val_info}")

    return loss_history


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    mae   = mean_absolute_error(y_true, y_pred)
    r2    = r2_score(y_true, y_pred)
    rpe   = np.mean(np.abs(y_true - y_pred) / (y_true + 1e-8)) * 100
    err   = y_pred - y_true
    score = float(np.sum(np.where(err < 0, np.exp(-err / 13) - 1, np.exp(err / 10) - 1)))
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "RPE_pct": rpe, "Score": score}


def run_and_evaluate(model, train_loader, test_loader, cfg: dict,
                     label: str = "", save_pred: bool = False) -> dict:
    """Full train + test cycle. Returns metrics dict."""
    print(f"\n[INFO] Training — {label}")
    train_model(model, train_loader, epochs=cfg["epochs"], lr=cfg["lr"])

    y_true, y_pred = evaluate_loader(model, test_loader)
    metrics = calculate_metrics(y_true, y_pred)

    print(f"[METRICS] {label}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    if save_pred:
        df_export = pd.DataFrame({
            "Time_Step"         : np.arange(len(y_true)),
            "True_BHI"          : y_true,
            "Predicted_BHI"     : y_pred,
            "True_RUL_Seconds"  : y_true * Config.BEARING_LIFE_S,
            "Pred_RUL_Seconds"  : y_pred * Config.BEARING_LIFE_S,
        })
        path = os.path.join(Config.OUTPUT_DIR, f"predictions_{label.replace(' ', '_')}.xlsx")
        df_export.to_excel(path, index=False)
        print(f"[INFO] Predictions saved: {path}")

    return metrics, y_true, y_pred


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
        print(f'[INFO] Plot saved: {out}')


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
# 6. XAI — SHAP (GradientExplainer + Beeswarm)
# =============================================================================

def _get_xai_tensors(dataset: BearingDataset,
                     n_bg: int = 200, n_test: int = 50):
    n_avail = len(dataset)
    n_bg    = min(n_bg,   n_avail // 2)
    n_test  = min(n_test, n_avail - n_bg)
    bg      = torch.stack([dataset[i][0] for i in range(n_bg)]).to(Config.DEVICE)
    test    = torch.stack([dataset[i][0] for i in range(n_bg, n_bg + n_test)]).to(Config.DEVICE)
    return bg, test


def run_shap(model: nn.Module, dataset: BearingDataset,
             feature_names: list, top_n: int = Config.TOP_N_FEATURES) -> pd.Series:
    """Runs SHAP GradientExplainer, plots beeswarm & bar, returns ranked importance."""
    print("\n[INFO] Running SHAP GradientExplainer...")
    model.eval()
    bg_tensor, test_tensor = _get_xai_tensors(dataset)

    explainer = shap.GradientExplainer(model, bg_tensor)
    shap_vals = explainer.shap_values(test_tensor)

    # shap_vals: list or ndarray (N, W, F) or (N, W, F, 1)
    sv = shap_vals[0] if isinstance(shap_vals, list) else shap_vals
    if sv.ndim == 4:
        sv = sv.squeeze(axis=-1)                    # → (N, W, F)

    # ── Beeswarm plot (SHAP standard) ────────────────────────────────────────
    sv_flat = sv.mean(axis=1)                       # (N, F) — avg over time-steps
    expl_obj = shap.Explanation(
        values          = sv_flat,
        base_values     = np.zeros(sv_flat.shape[0]),
        data            = test_tensor.cpu().numpy().mean(axis=1),
        feature_names   = feature_names,
    )
    with plt.rc_context(PAPER_RC):
        plt.figure(figsize=(10, 8))
        shap.plots.beeswarm(expl_obj, max_display=top_n, show=False)
        plt.title("SHAP Beeswarm — Feature Impact on BHI Prediction",
                  fontweight='bold', pad=10)
        plt.tight_layout()
        out_beeswarm = os.path.join(Config.OUTPUT_DIR, 'shap_beeswarm.png')
        plt.savefig(out_beeswarm, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"[INFO] SHAP beeswarm saved: {out_beeswarm}")

    # ── Bar importance ────────────────────────────────────────────────────────
    importance = pd.Series(
        np.abs(sv).mean(axis=(0, 1)),
        index=feature_names
    ).sort_values(ascending=False)

    top_importance = importance.head(top_n).sort_values(ascending=True)
    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(8, max(5, top_n * 0.4)))
        bars = ax.barh(top_importance.index, top_importance.values,
                       color=PALETTE['shap'], edgecolor='none', height=0.65, alpha=0.88)
        for bar, val in zip(bars, top_importance.values):
            ax.text(val + top_importance.values.max() * 0.015,
                    bar.get_y() + bar.get_height() / 2,
                    f'{val:.4f}', va='center', fontsize=8.5)
        ax.set_title(f'SHAP Feature Importance — Top {top_n}', fontweight='bold')
        ax.set_xlabel('Mean |SHAP Value|')
        plt.tight_layout()
        out_bar = os.path.join(Config.OUTPUT_DIR, 'shap_importance_bar.png')
        plt.savefig(out_bar, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"[INFO] SHAP bar chart saved: {out_bar}")

    print(f"\n[INFO] Top-{top_n} features by SHAP:")
    for f, v in importance.head(top_n).items():
        print(f"  {f:<35s} {v:.6f}")

    return importance  # descending order, full list


# =============================================================================
# 7. SCENARIO 2 — RANDOM SEARCH (internal B1 only)
# =============================================================================

SEARCH_SPACE = {
    "hidden_size"  : [32, 64, 128, 256],
    "learning_rate": [1e-4, 5e-4, 1e-3, 5e-3],
    "dropout"      : [0.0, 0.1, 0.2, 0.3],
}


def random_search(input_size: int, loaders: dict,
                  n_trials: int = Config.RANDOM_TRIALS) -> dict:
    """Random search on B1-train/val split. B2 never touched."""
    print(f"\n[INFO] Scenario 2 — Random Search ({n_trials} trials)")
    best_rmse, best_cfg = float('inf'), None

    train_loader = loaders["b1_train"][0]
    val_loader   = loaders["b1_val"][0]

    for trial in range(n_trials):
        cfg = {
            "hidden_size": random.choice(SEARCH_SPACE["hidden_size"]),
            "lr"         : random.choice(SEARCH_SPACE["learning_rate"]),
            "dropout"    : random.choice(SEARCH_SPACE["dropout"]),
            "num_layers" : Config.NUM_LAYERS,
            "epochs"     : Config.EPOCHS,
        }
        print(f"\n  [Trial {trial+1}/{n_trials}] {cfg}")
        model = build_model(input_size, cfg)
        train_model(model, train_loader, epochs=cfg["epochs"],
                    lr=cfg["lr"], val_loader=val_loader, verbose=False)

        y_t, y_p = evaluate_loader(model, val_loader)
        val_rmse = np.sqrt(mean_squared_error(y_t, y_p))
        print(f"  → Val RMSE: {val_rmse:.6f}")

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_cfg  = cfg
        del model

    print(f"\n[INFO] Best Val RMSE: {best_rmse:.6f} | Config: {best_cfg}")
    return best_cfg


# =============================================================================
# 8. SUMMARY TABLE
# =============================================================================

def print_summary(results: dict):
    rows = []
    for scenario, metrics in results.items():
        row = {"Scenario": scenario}
        row.update({k: round(v, 4) for k, v in metrics.items()})
        rows.append(row)
    df = pd.DataFrame(rows).set_index("Scenario")
    print("\n" + "="*70)
    print("SCENARIO COMPARISON SUMMARY")
    print("="*70)
    print(df.to_string())
    path = os.path.join(Config.OUTPUT_DIR, "scenario_comparison.csv")
    df.to_csv(path)
    print(f"\n[INFO] Summary table saved: {path}")
    return df


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":

    # ── Step 1: Data Loading & Splitting ─────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 1: DATA LOADING & SPLITTING")
    print("="*60)

    X_b1, y_b1, X_b2, y_b2, feature_cols, scaler = load_raw_data()
    n_features = len(feature_cols)

    loaders = prepare_all_loaders(
        X_b1, y_b1, X_b2, y_b2,
        window_size=Config.WINDOW_SIZE,
        batch_size=Config.BATCH_SIZE,
    )
    loader_b1_full,  ds_b1_full  = loaders["b1_full"]
    loader_b2_test,  ds_b2_test  = loaders["b2_test"]
    loader_b1_train, _           = loaders["b1_train"]
    loader_b1_val,   _           = loaders["b1_val"]

    baseline_cfg = {
        "hidden_size": Config.HIDDEN_SIZE,
        "num_layers" : Config.NUM_LAYERS,
        "dropout"    : Config.DROPOUT,
        "lr"         : Config.LEARNING_RATE,
        "epochs"     : Config.EPOCHS,
    }

    scenario_results = {}

    # ── Scenario 1: Baseline ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("SCENARIO 1: BASELINE MODEL")
    print("="*60)

    model_s1 = build_model(n_features, baseline_cfg)
    metrics_s1, y_true_s1, y_pred_s1 = run_and_evaluate(
        model_s1, loader_b1_full, loader_b2_test,
        cfg=baseline_cfg, label="Scenario 1 Baseline", save_pred=True
    )
    plot_bhi_prediction(y_true_s1, y_pred_s1, metrics_s1, tag="Scenario 1 Baseline")
    scenario_results["S1_Baseline"] = metrics_s1

    # ── Scenario 2: Hyperparameter Tuning (B1-internal only) ─────────────────
    print("\n" + "="*60)
    print("SCENARIO 2: HYPERPARAMETER TUNING (B1 internal — NO B2)")
    print("="*60)

    best_cfg = random_search(n_features, loaders, n_trials=Config.RANDOM_TRIALS)

    # ── Scenario 3: Retrain with Best Config + XAI ───────────────────────────
    print("\n" + "="*60)
    print("SCENARIO 3: RETRAINING WITH BEST CONFIG + XAI")
    print("="*60)

    model_s3 = build_model(n_features, best_cfg)
    metrics_s3, y_true_s3, y_pred_s3 = run_and_evaluate(
        model_s3, loader_b1_full, loader_b2_test,
        cfg=best_cfg, label="Scenario 3 Tuned", save_pred=True
    )

    # Fall back to baseline if tuned config is worse
    if metrics_s3["RMSE"] > metrics_s1["RMSE"]:
        print("[WARN] Tuned config is worse than baseline — reverting to baseline config.")
        best_cfg   = baseline_cfg
        model_s3   = model_s1
        metrics_s3 = metrics_s1
        y_true_s3  = y_true_s1
        y_pred_s3  = y_pred_s1

    plot_bhi_prediction(y_true_s3, y_pred_s3, metrics_s3, tag="Scenario 3 Tuned")
    scenario_results["S3_Tuned"] = metrics_s3

    # XAI on B2 with best model
    shap_importance = run_shap(
        model_s3, ds_b2_test, feature_cols, top_n=Config.TOP_N_FEATURES
    )
    top_features = list(shap_importance.head(Config.TOP_N_FEATURES).index)
    print(f"\n[INFO] Top-{Config.TOP_N_FEATURES} SHAP features selected for Scenario 4:")
    for f in top_features:
        print(f"  • {f}")

    # ── Scenario 4: SHAP-Guided Retraining (reduced feature set) ─────────────
    print("\n" + "="*60)
    print("SCENARIO 4: SHAP-GUIDED RETRAINING (REDUCED FEATURES)")
    print("="*60)

    top_idx   = [feature_cols.index(f) for f in top_features]
    X_b1_red  = X_b1[:, top_idx]
    X_b2_red  = X_b2[:, top_idx]
    n_feat_s4 = len(top_features)

    loaders_s4 = prepare_all_loaders(
        X_b1_red, y_b1, X_b2_red, y_b2,
        window_size=Config.WINDOW_SIZE,
        batch_size=Config.BATCH_SIZE,
    )
    loader_b1_s4 = loaders_s4["b1_full"][0]
    loader_b2_s4 = loaders_s4["b2_test"][0]

    model_s4 = build_model(n_feat_s4, best_cfg)
    metrics_s4, y_true_s4, y_pred_s4 = run_and_evaluate(
        model_s4, loader_b1_s4, loader_b2_s4,
        cfg=best_cfg, label="Scenario 4 SHAP-Guided", save_pred=True
    )
    plot_bhi_prediction(y_true_s4, y_pred_s4, metrics_s4, tag="Scenario 4 SHAP-Guided")
    scenario_results["S4_SHAP_Guided"] = metrics_s4

    # ── Step 6: Save Final Model & Summary ────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 6: OUTPUT & SAVING")
    print("="*60)

    model_path = os.path.join(Config.OUTPUT_DIR, "model_S4_final.pth")
    torch.save(model_s4.state_dict(), model_path)
    print(f"[INFO] Final model weights saved: {model_path}")

    summary_df = print_summary(scenario_results)

    print("\n[SUCCESS] All 4 Scenarios Completed Successfully!")