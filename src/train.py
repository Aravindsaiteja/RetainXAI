"""
Neural Network Model Training and Experiment Tracking Module.
Implements PyTorch training loop with BCEWithLogitsLoss (weighted for class imbalance),
AdamW optimizer, Learning Rate Scheduler, Early Stopping, Model Checkpointing,
Hyperparameter Experimentation, and Training Metrics Visualization.
"""

import sys
from pathlib import Path
import copy
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.model import TabularChurnNN


def set_seed(seed: int = config.RANDOM_SEED):
    """Sets random seeds for reproducibility across numpy, torch CPU/CUDA."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_dataloaders(
    train_path: Path = config.PROCESSED_TRAIN_FILE,
    val_path: Path = config.PROCESSED_VAL_FILE,
    batch_size: int = 64,
) -> tuple:
    """Loads processed CSVs and creates PyTorch DataLoaders."""
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)

    X_train = df_train.drop(columns=[config.TARGET_COLUMN]).values.astype(np.float32)
    y_train = df_train[config.TARGET_COLUMN].values.astype(np.float32).reshape(-1, 1)

    X_val = df_val.drop(columns=[config.TARGET_COLUMN]).values.astype(np.float32)
    y_val = df_val[config.TARGET_COLUMN].values.astype(np.float32).reshape(-1, 1)

    # Class imbalance weight
    pos_count = np.sum(y_train == 1)
    neg_count = np.sum(y_train == 0)
    pos_weight = float(neg_count / pos_count)

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, X_train.shape[1], pos_weight


def train_single_experiment(
    exp_name: str,
    hidden_dim_1: int = 64,
    hidden_dim_2: int = 32,
    dropout_rate: float = 0.25,
    learning_rate: float = 0.001,
    batch_size: int = 64,
    epochs: int = 100,
    patience: int = 10,
    device: torch.device = config.DEVICE,
) -> tuple:
    """
    Executes a complete training and validation cycle with Early Stopping.
    """
    set_seed(config.RANDOM_SEED)
    train_loader, val_loader, input_dim, pos_weight = create_dataloaders(batch_size=batch_size)

    # Instantiate model
    model = TabularChurnNN(
        input_dim=input_dim,
        hidden_dim_1=hidden_dim_1,
        hidden_dim_2=hidden_dim_2,
        dropout_rate=dropout_rate,
    ).to(device)

    # Loss & Optimizer
    pos_weight_tensor = torch.tensor([pos_weight], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_f1": [],
        "val_roc_auc": [],
    }

    best_val_loss = float("inf")
    best_model_weights = None
    patience_counter = 0
    best_metrics = {}

    print(f"\n--- Starting Experiment: [{exp_name}] ---")
    print(f"Arch: [{input_dim} -> {hidden_dim_1} -> {hidden_dim_2} -> 1] | LR: {learning_rate} | Dropout: {dropout_rate}")

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        train_loss_sum = 0.0
        train_preds, train_targets = [], []

        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)

            optimizer.zero_grad()
            logits = model(X_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * X_b.size(0)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            train_preds.extend((probs >= 0.5).astype(int))
            train_targets.extend(y_b.cpu().numpy())

        train_loss = train_loss_sum / len(train_loader.dataset)
        train_acc = accuracy_score(train_targets, train_preds)

        # Validation Phase
        model.eval()
        val_loss_sum = 0.0
        val_probs, val_targets = [], []

        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                logits = model(X_b)
                loss = criterion(logits, y_b)
                val_loss_sum += loss.item() * X_b.size(0)

                probs = torch.sigmoid(logits).cpu().numpy()
                val_probs.extend(probs)
                val_targets.extend(y_b.cpu().numpy())

        val_loss = val_loss_sum / len(val_loader.dataset)
        val_probs = np.array(val_probs).ravel()
        val_targets = np.array(val_targets).ravel()
        val_preds = (val_probs >= 0.5).astype(int)

        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, zero_division=0)
        val_roc_auc = roc_auc_score(val_targets, val_probs)

        # Scheduler step
        scheduler.step(val_loss)

        # Record history
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["val_roc_auc"].append(val_roc_auc)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | Val AUC: {val_roc_auc:.4f}")

        # Checkpoint Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
            best_metrics = {
                "best_epoch": epoch,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "val_roc_auc": val_roc_auc,
            }
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[EARLY STOPPING] Triggered at epoch {epoch}. Best Val Loss: {best_val_loss:.4f} (Epoch {best_metrics['best_epoch']})")
                break

    elapsed = time.time() - start_time
    print(f"[COMPLETED] {exp_name} in {elapsed:.2f}s. Best Epoch: {best_metrics['best_epoch']} | Val Acc: {best_metrics['val_acc']:.4f} | Val F1: {best_metrics['val_f1']:.4f} | Val AUC: {best_metrics['val_roc_auc']:.4f}")

    # Load best weights into model
    model.load_state_dict(best_model_weights)
    return model, history, best_metrics


def run_experiment_tracking_and_training() -> tuple:
    """
    Executes multiple hyperparameter experiments, logs performance table,
    saves best model checkpoint to models/nn_churn_model.pt, and plots training curves.
    """
    print("\n" + "=" * 65)
    print("        EXPERIMENT TRACKING & NEURAL NETWORK TRAINING")
    print("=" * 65)

    experiments_config = [
        {
            "exp_name": "Exp1_Baseline_64_32",
            "hidden_dim_1": 64,
            "hidden_dim_2": 32,
            "dropout_rate": 0.25,
            "learning_rate": 0.001,
        },
        {
            "exp_name": "Exp2_Wider_128_64",
            "hidden_dim_1": 128,
            "hidden_dim_2": 64,
            "dropout_rate": 0.30,
            "learning_rate": 0.001,
        },
        {
            "exp_name": "Exp3_LowerLR_64_32",
            "hidden_dim_1": 64,
            "hidden_dim_2": 32,
            "dropout_rate": 0.20,
            "learning_rate": 0.0005,
        },
    ]

    exp_records = []
    best_overall_auc = -1.0
    best_overall_model = None
    best_overall_history = None
    best_exp_info = None

    for cfg in experiments_config:
        model, history, best_metrics = train_single_experiment(
            exp_name=cfg["exp_name"],
            hidden_dim_1=cfg["hidden_dim_1"],
            hidden_dim_2=cfg["hidden_dim_2"],
            dropout_rate=cfg["dropout_rate"],
            learning_rate=cfg["learning_rate"],
        )

        arch_str = f"[{cfg['hidden_dim_1']}-{cfg['hidden_dim_2']}]"
        record = {
            "Experiment": cfg["exp_name"],
            "Architecture": arch_str,
            "Epochs_Trained": len(history["epoch"]),
            "Best_Epoch": best_metrics["best_epoch"],
            "Learning_Rate": cfg["learning_rate"],
            "Val_Loss": round(best_metrics["val_loss"], 4),
            "Accuracy": round(best_metrics["val_acc"], 4),
            "F1_Score": round(best_metrics["val_f1"], 4),
            "ROC_AUC": round(best_metrics["val_roc_auc"], 4),
        }
        exp_records.append(record)

        if best_metrics["val_roc_auc"] > best_overall_auc:
            best_overall_auc = best_metrics["val_roc_auc"]
            best_overall_model = model
            best_overall_history = history
            best_exp_info = cfg

    # Save Experiment Tracking Table
    exp_df = pd.DataFrame(exp_records)
    exp_df.to_csv(config.EXPERIMENT_LOG_FILE, index=False)
    print("\n" + "=" * 65)
    print("               EXPERIMENT TRACKING COMPARISON TABLE")
    print("=" * 65)
    print(exp_df.to_string(index=False))
    print(f"\n[WINNING EXPERIMENT]: '{best_exp_info['exp_name']}' (Validation ROC-AUC: {best_overall_auc:.4f})")

    # Save Winning Model Checkpoint
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_overall_model.state_dict(), config.MODEL_FILE)
    print(f"[SAVED] Winning Model Checkpoint -> {config.MODEL_FILE}")

    # Plot Training Curves for Winning Model
    plot_training_history(best_overall_history, best_exp_info["exp_name"])

    return best_overall_model, best_overall_history, exp_df


def plot_training_history(history: dict, exp_name: str):
    """Plots and saves Training & Validation Loss/Accuracy curves."""
    figures_dir = config.REPORTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    epochs = history["epoch"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Loss Curve
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#2b5c8f", linewidth=2)
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="#d9534f", linewidth=2, linestyle="--")
    ax1.set_title(f"Training & Validation Loss ({exp_name})", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Weighted BCE Loss", fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Accuracy & ROC-AUC Curve
    ax2.plot(epochs, history["val_acc"], label="Val Accuracy", color="#27ae60", linewidth=2)
    ax2.plot(epochs, history["val_roc_auc"], label="Val ROC-AUC", color="#8e44ad", linewidth=2, linestyle="--")
    ax2.plot(epochs, history["val_f1"], label="Val F1-Score", color="#e67e22", linewidth=2, linestyle=":")
    ax2.set_title(f"Validation Metrics Progress ({exp_name})", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Score", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    curve_path = figures_dir / "06_training_curves.png"
    plt.savefig(curve_path, dpi=300)
    plt.close()
    print(f"[SAVED] Training Curves Plot -> {curve_path}\n")


if __name__ == "__main__":
    run_experiment_tracking_and_training()
