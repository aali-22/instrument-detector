import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import SimpleAudioCNN
from dataset import MelSpecDataset
import os
import csv
import time
import logging
from pathlib import Path

# =====================
# Utility Functions
# =====================

def setup_logging(log_dir):
    """Set up logging to file and console."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'train.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging to {log_file}")

def cuda_sanity_check():
    """Check CUDA availability and print GPU info."""
    logging.info(f"PyTorch version: {torch.__version__}")
    cuda_ok = torch.cuda.is_available()
    logging.info(f"CUDA available: {cuda_ok}")
    if cuda_ok:
        logging.info(f"GPU name: {torch.cuda.get_device_name(0)}")
        logging.info(f"CUDA version: {torch.version.cuda}")
    else:
        logging.warning("No GPU detected by PyTorch. Training will use CPU and may be slow.")
    return cuda_ok

def validate_paths(paths_dict):
    """Validate all required paths exist."""
    missing_paths = []
    for name, path in paths_dict.items():
        if not os.path.exists(path):
            missing_paths.append(f"{name}: {path}")
    
    if missing_paths:
        logging.error("Missing required paths:")
        for path in missing_paths:
            logging.error(f"  - {path}")
        return False
    return True

def save_csv_log(log_path, log_rows):
    """Save training and validation metrics to a CSV file."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        writer.writerows(log_rows)
    logging.info(f"Training log saved to: {log_path}")

def test_save_functionality(model, save_path):
    """Test if model saving works correctly."""
    try:
        torch.save(model.state_dict(), save_path)
        loaded_model = SimpleAudioCNN(n_mels=128, n_classes=4)
        loaded_model.load_state_dict(torch.load(save_path))
        logging.info(f"✓ Model save/load test successful at {save_path}")
        return True
    except Exception as e:
        logging.error(f"✗ Model save/load test failed: {str(e)}")
        return False

def auto_checkpoint(model, checkpoint_dir, epoch, val_acc):
    """Save a checkpoint every N epochs."""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    checkpoint_path = os.path.join(checkpoint_dir, f"model_e{epoch}_acc{val_acc*100:.1f}_{timestamp}.pt")
    torch.save(model.state_dict(), checkpoint_path)
    logging.info(f"[Checkpoint] Model saved to {checkpoint_path}")

# =====================
# Training/Eval Loops
# =====================
def train_one_epoch(model, loader, criterion, optimizer, device):
    """Run one training epoch over the provided DataLoader."""
    model.train()
    running_loss, running_correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        preds = out.argmax(dim=1)
        running_loss += loss.item() * xb.size(0)
        running_correct += (preds == yb).sum().item()
        total += xb.size(0)
    avg_loss = running_loss / total
    avg_acc = running_correct / total
    return avg_loss, avg_acc

def eval_one_epoch(model, loader, criterion, device):
    """Evaluate the model on the provided DataLoader."""
    model.eval()
    running_loss, running_correct, total = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            preds = out.argmax(dim=1)
            running_loss += loss.item() * xb.size(0)
            running_correct += (preds == yb).sum().item()
            total += xb.size(0)
    avg_loss = running_loss / total
    avg_acc = running_correct / total
    return avg_loss, avg_acc

# =====================
# Main Training Script
# =====================
def main():
    # --- Set up output directories ---
    base_path = Path(__file__).parent.parent  # Get project root directory
    run_timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = base_path / "outputs" / f"run_{run_timestamp}"
    checkpoint_dir = run_dir / "checkpoints"
    log_path = run_dir / "train_log.csv"
    best_model_path = run_dir / "best_model.pt"

    # Create directories
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # --- Set up logging ---
    setup_logging(run_dir)
    logging.info(f"Project root: {base_path}")
    logging.info(f"Run directory: {run_dir}")

    # --- CUDA sanity check ---
    cuda_sanity_check()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Training configuration ---
    train_mel_dir = base_path / "data" / "mel_spectrograms" / "train"
    train_json = base_path / "data" / "json" / "nsynth-train-filtered.json"
    val_mel_dir = base_path / "data" / "mel_spectrograms" / "valid"
    val_json = base_path / "data" / "json" / "nsynth-valid-filtered.json"
    
    # Validate all required paths
    paths_to_check = {
        "Training mel spectrograms": train_mel_dir,
        "Training JSON": train_json,
        "Validation mel spectrograms": val_mel_dir,
        "Validation JSON": val_json
    }
    
    if not validate_paths(paths_to_check):
        logging.error("Missing required data files. Please check the paths above.")
        return

    # Training hyperparameters
    batch_size = 32
    n_classes = 4
    n_mels = 128
    epochs = 50
    patience = 10
    checkpoint_every = 5  # Save checkpoint every N epochs

    # --- Initialize model and test save functionality ---
    model = SimpleAudioCNN(n_mels=n_mels, n_classes=n_classes).to(device)
    if not test_save_functionality(model, best_model_path):
        logging.error("Model save test failed. Please check your file permissions and paths.")
        return

    # --- Prepare datasets and data loaders ---
    train_ds = MelSpecDataset(train_mel_dir, train_json)
    val_ds = MelSpecDataset(val_mel_dir, val_json, label_map=train_ds.label_map)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # --- Initialize loss and optimizer ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_val_acc = 0.0
    best_epoch = 0
    log_rows = []
    patience_counter = 0

    logging.info("Starting training...")
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_dl, criterion, optimizer, device)
        val_loss, val_acc = eval_one_epoch(model, val_dl, criterion, device)
        log_rows.append([epoch, train_loss, train_acc, val_loss, val_acc])
        logging.info(f"Epoch {epoch:3d}: Train Loss={train_loss:.4f} Acc={train_acc*100:.1f}% | Val Loss={val_loss:.4f} Acc={val_acc*100:.1f}%")

        # Save model if validation accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"[Best] Model saved to {best_model_path}")
            auto_checkpoint(model, checkpoint_dir, epoch, val_acc)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break
        # Auto-checkpoint every N epochs
        if epoch % checkpoint_every == 0:
            auto_checkpoint(model, checkpoint_dir, epoch, val_acc)

    save_csv_log(log_path, log_rows)
    logging.info(f"Training complete. Best val acc: {best_val_acc*100:.1f}% at epoch {best_epoch}")
    logging.info(f"Best model saved to: {best_model_path}")

if __name__ == "__main__":
    main()
