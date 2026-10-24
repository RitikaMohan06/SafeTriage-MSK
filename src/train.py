"""
train.py
========
Standard binary classification training loop with class-weighted loss
(MURA is mildly imbalanced towards normal studies in some body parts),
and early stopping on the TUNE split's validation loss — never on
calibration or test, to avoid leaking information into calibration.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def compute_pos_weight(train_df):
    """Weight the loss so rare classes aren't ignored. Ratio of neg:pos."""
    n_pos = (train_df.label == 1).sum()
    n_neg = (train_df.label == 0).sum()
    return torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float32)


def train_model(model, train_df, tune_df, DatasetClass, epochs=10, batch_size=16,
                 lr=1e-4, device="cuda", patience=3, out_path="outputs/best_model.pt"):
    device = device if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    train_ds = DatasetClass(train_df, train=True)
    tune_ds = DatasetClass(tune_df, train=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    tune_loader = DataLoader(tune_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    pos_weight = compute_pos_weight(train_df).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=1)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for imgs, labels, _, _ in tqdm(train_loader, desc=f"Epoch {epoch+1} [train]"):
            imgs, labels = imgs.to(device), labels.float().to(device)
            optimizer.zero_grad()
            logits, _ = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels, _, _ in tune_loader:
                imgs, labels = imgs.to(device), labels.float().to(device)
                logits, _ = model(imgs)
                loss = criterion(logits, labels)
                val_loss += loss.item() * imgs.size(0)
        val_loss /= len(tune_ds)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        # Early stopping keeps us from overfitting to the train split, and
        # ensures the weights we later calibrate/evaluate are the best
        # generalizing ones, not just the last epoch's.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), out_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping.")
                break

    model.load_state_dict(torch.load(out_path))
    return model
