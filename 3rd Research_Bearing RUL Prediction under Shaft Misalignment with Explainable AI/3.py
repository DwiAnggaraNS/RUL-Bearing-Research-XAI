# Generated from: 3.ipynb
# Converted at: 2026-06-03T11:36:16.263Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# Bearing RUL Prediction: Single LSTM Model with SHAP & Integrated Gradients
# Research: Cross-Domain Bearing RUL Prediction under Shaft Misalignment


import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["OMP_NUM_THREADS"] = "1" 
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# import shap
# from captum.attr import IntegratedGradients


torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True

# # 1. GLOBAL CONFIGURATION


# 1. PERBAIKI KOLOM YANG DIBUANG
class Config:
    TRAIN_DATA_PATH = r"D:\ProyekDosen\RisetBearing\bearing_1\processed_bearing1.parquet"
    TEST_DATA_PATH  = r"D:\ProyekDosen\RisetBearing\bearing_2\processed_bearing2_synthetic.parquet"
    OUTPUT_DIR      = r"D:\ProyekDosen\RisetBearing\results"
        
    COLS_TO_DROP = ['segment', 'time_s', 'time_min', 'bhi', 'label_vcd', 'T_cp', 'T_f',
                    'rms_ema_x', 'rms_ema_y', 'rms_ema_z']  # tambahkan ema cols juga
    TARGET_COL   = 'bhi'   # pastikan match exact dengan nama kolom parquet

    WINDOW_SIZE     = 60
    BATCH_SIZE      = 32
    HIDDEN_SIZE     = 64
    NUM_LAYERS      = 2
    LEARNING_RATE   = 0.001
    EPOCHS          = 50
    # DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DEVICE          = torch.device("cpu")
    BEARING_LIFE_S  = 392275

# Create output directory if it doesn't exist
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
print(f"Device: {Config.DEVICE}")

torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True

# # 2. DATA STRUCTURES & LOADERS


class BearingDataset(Dataset):
    def __init__(self, X, y, window_size, stride=1):
        self.X = X
        self.y = y
        self.window_size = window_size
        self.stride = stride
        self.indices = list(range(0, len(X) - window_size, stride))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        end = start + self.window_size
        return torch.tensor(self.X[start:end], dtype=torch.float32), \
               torch.tensor([self.y[end-1]], dtype=torch.float32)

def load_and_preprocess_data():
    """Loads parquet, scales features, and creates sequence loaders"""
    print("[INFO] Loading datasets...")
    df_train = pd.read_parquet(Config.TRAIN_DATA_PATH)
    df_test  = pd.read_parquet(Config.TEST_DATA_PATH)
    
    # Identify feature columns
    drop_cols = Config.COLS_TO_DROP + [Config.TARGET_COL]
    feature_cols = [c for c in df_train.columns if c not in drop_cols]
    print(f"[INFO] Detected {len(feature_cols)} input features.")
    
    # Extract Target
    y_train = df_train[Config.TARGET_COL].values
    y_test  = df_test[Config.TARGET_COL].values
    
    # Scale Features (Fit on TRAIN ONLY to prevent data leakage)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(df_train[feature_cols].values)
    X_test  = scaler.transform(df_test[feature_cols].values)
    
    # Create DataLoaders
    train_dataset = BearingDataset(X_train, y_train, Config.WINDOW_SIZE, stride=1)
    test_dataset  = BearingDataset(X_test, y_test, Config.WINDOW_SIZE, stride=1)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader, feature_cols, train_dataset, test_dataset

# # 3. MODEL ARCHITECTURE
# 


class RULLSTM(nn.Module):
    def __init__(self, input_size: int):
        super(RULLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=64,  # Increased from 64
            num_layers=2,      # Increased from 2
            batch_first=True,
            dropout=0.0        # Increased dropout
        )
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_timestep = lstm_out[:, -1, :]
        out = self.relu(self.fc1(last_timestep))
        out = self.sigmoid(self.fc2(out))
        return out

# 
# # 4. TRAINING & EVALUATION FUNCTIONS
# 


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Computes RUL/BHI evaluation metrics"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Relative Prediction Error (RPE) & Scoring Function
    epsilon = 1e-8
    rpe = np.mean(np.abs(y_true - y_pred) / (y_true + epsilon)) * 100
    
    # PRONOSTIA Asymmetric Scoring Function
    errors = y_pred - y_true
    score = np.sum(np.where(errors < 0, np.exp(-errors/13) - 1, np.exp(errors/10) - 1))
    
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'RPE_pct': rpe, 'Score': score}

def train_model(model, train_loader):
    criterion = nn.MSELoss(reduction='none')  # Changed to no reduction
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)  # Higher LR
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    loss_history = []
    print("\n[INFO] Starting Training...")
    
    for epoch in range(Config.EPOCHS):
        model.train()
        epoch_losses = []
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(Config.DEVICE), y_batch.to(Config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            
            # Weighted loss - give more weight to degradation phase
            weights = torch.where(
                y_batch < 0.98,
                torch.tensor(15.0).to(Config.DEVICE),
                torch.tensor(1.0).to(Config.DEVICE)
            )
            loss = (criterion(outputs, y_batch) * weights).mean()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
            
        avg_loss = np.mean(epoch_losses)
        loss_history.append(avg_loss)
        scheduler.step(avg_loss)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{Config.EPOCHS}] | Loss: {avg_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
            
    return loss_history

def evaluate_and_export(model, test_loader):
    """Evaluates the model and exports predictions to Excel"""
    model.eval()
    predictions, targets = [], []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(Config.DEVICE)
            preds = model(X_batch).cpu().numpy()
            predictions.extend(preds.flatten())
            targets.extend(y_batch.numpy().flatten())
            
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    metrics = calculate_metrics(targets, predictions)
    print("\n[INFO] Evaluation Metrics on Bearing-2 (Test Set):")
    for k, v in metrics.items():
        print(f" - {k}: {v:.4f}")
        
    # Convert BHI to RUL (Simple linear conversion for evaluation)
    true_rul = targets * Config.BEARING_LIFE_S
    pred_rul = predictions * Config.BEARING_LIFE_S
    
    # Export to Excel
    df_export = pd.DataFrame({
        'Time_Step': np.arange(len(targets)),
        'True_BHI': targets,
        'Predicted_BHI': predictions,
        'True_RUL_Seconds': true_rul,
        'Predicted_RUL_Seconds': pred_rul
    })
    
    export_path = os.path.join(Config.OUTPUT_DIR, "Predictions_Results.xlsx")
    df_export.to_excel(export_path, index=False)
    print(f"[INFO] Predictions saved successfully to: {export_path}")
    
    return targets, predictions

# # 5. VISUALIZATIONS & XAI (SHAP & CAPTUM)
# 


def plot_rul(y_true: np.ndarray, y_pred: np.ndarray):
    """Plots actual vs predicted Health Index"""
    plt.figure(figsize=(12, 5))
    plt.plot(y_true, label='True Health Index (BHI)', color='blue', linewidth=2)
    plt.plot(y_pred, label='Predicted Health Index', color='red', linestyle='--', linewidth=2)
    plt.title('Bearing Health Index (BHI) Prediction on Shaft Misalignment Data')
    plt.xlabel('Time Steps')
    plt.ylabel('Health Index (1 = Healthy, 0 = Failed)')
    plt.legend()
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(Config.OUTPUT_DIR, "BHI_Prediction_Plot.png"), dpi=300)
    plt.show()

# def run_xai(model, test_dataset, feature_names):
#     """Applies SHAP and Integrated Gradients for Model Interpretability"""
#     print("\n[INFO] Initializing Explainable AI (XAI) Analysis...")
#     model.eval()
    
#     # Sample background and test instances for XAI (Memory optimization)
#     background_tensor = test_dataset.X[:100].to(Config.DEVICE)
#     test_tensor = test_dataset.X[100:105].to(Config.DEVICE)
    
#     # --- 1. CAPTUM (Integrated Gradients) ---
#     print("[INFO] Executing Integrated Gradients (Captum)...")
#     ig = IntegratedGradients(model)
#     attributions, delta = ig.attribute(test_tensor, target=None, return_convergence_delta=True)
    
#     # Summarize attributions across the time window (dim 1)
#     attr_sum = attributions.cpu().detach().numpy().sum(axis=1).mean(axis=0)
    
#     # --- 2. SHAP (GradientExplainer) ---
#     print("[INFO] Executing SHAP GradientExplainer...")
#     explainer = shap.GradientExplainer(model, background_tensor)
#     shap_values = explainer.shap_values(test_tensor)
    
#     # SHAP returns a list of arrays for PyTorch, we need to average across timesteps
#     shap_val_2d = shap_values[0].mean(axis=1) if isinstance(shap_values, list) else shap_values.mean(axis=1)
    
#     # --- Visualizing XAI ---
#     plt.figure(figsize=(10, 8))
#     # Plot top 15 features based on Integrated Gradients
#     feat_importances = pd.Series(np.abs(attr_sum), index=feature_names)
#     feat_importances.nlargest(15).sort_values().plot(kind='barh', color='teal')
#     plt.title("Top 15 Dominant Features (Integrated Gradients)")
#     plt.xlabel("Absolute Attribution Magnitude")
#     plt.tight_layout()
#     plt.savefig(os.path.join(Config.OUTPUT_DIR, "XAI_IntegratedGradients_Plot.png"), dpi=300)
#     plt.show()

# # MAIN EXECUTION
# 


if __name__ == "__main__":
    # 1. Load Data
    train_loader, test_loader, features, _, test_ds = load_and_preprocess_data()
    
    # 2. Initialize Model
    model = RULLSTM(input_size=len(features)).to(Config.DEVICE)
    
    # 3. Train
    loss_hist = train_model(model, train_loader)
    
    # 4. Evaluate & Export
    y_true, y_pred = evaluate_and_export(model, test_loader)
    plot_rul(y_true, y_pred)
    
    # 5. Interpret (XAI)
    # run_xai(model, test_ds, features)
    
    print("\n[SUCCESS] Pipeline Executed Successfully!")