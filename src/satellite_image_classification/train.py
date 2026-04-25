import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .config import (
    BATCH_SIZE,
    DATASET_DIR,
    DEVICE,
    EPOCHS,
    EXPERIMENTS_DIR,
    LEARNING_RATE,
    N_SPLITS,
    NUM_WORKERS,
    RUNS_DIR,
    SEED,
)
from .dataset import SatelliteDataset, get_image_label_pairs, get_transforms
from .model import SatelliteCNN
from .utils import (
    get_accuracy,
    get_classification_metrics,
    plot_confusion_matrix,
    plot_history,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_fold(model, train_loader, val_loader, criterion, optimizer, epochs, name):
    print(f"\nStarting {name}...")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    log_dir = RUNS_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{name}"
    writer = SummaryWriter(str(log_dir))
    results_dir = EXPERIMENTS_DIR / name
    results_dir.mkdir(parents=True, exist_ok=True)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_acc = 0.0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_acc += get_accuracy(outputs, labels) * imgs.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = running_acc / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        model.eval()
        val_running_loss = 0.0
        val_running_acc = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item()
                val_running_acc += get_accuracy(outputs, labels) * imgs.size(0)

                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = val_running_loss / len(val_loader)
        val_acc = val_running_acc / len(val_loader.dataset)
        precision, recall, f1 = get_classification_metrics(all_labels, all_preds)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(
            f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f} "
            f"| Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f} "
            f"- Val Precision: {precision:.4f} - Val Recall: {recall:.4f} - Val F1: {f1:.4f}"
        )

        writer.add_scalar("Loss/train", epoch_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/train", epoch_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.add_scalar("Metrics/precision_macro", precision, epoch)
        writer.add_scalar("Metrics/recall_macro", recall, epoch)
        writer.add_scalar("Metrics/f1_macro", f1, epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(results_dir, "best_model.pth"))

        plot_confusion_matrix(all_labels, all_preds, epoch + 1, str(results_dir))

    plot_history(train_losses, val_losses, train_accs, val_accs, str(results_dir))
    writer.close()
    return best_acc


def main():
    set_seed(SEED)
    print(f"Using device: {DEVICE}")
    print(f"Using seed: {SEED}")

    print("Loading data...")
    data = get_image_label_pairs(str(DATASET_DIR))
    if not data:
        raise ValueError(f"No images found in '{DATASET_DIR}'. Check folder structure and file extensions.")
    labels = [label for _, label in data]

    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_accuracies = []

    print(f"Starting {N_SPLITS}-Fold Stratified Cross Validation...")

    for fold, (train_idx, val_idx) in enumerate(splitter.split(data, labels), start=1):
        print(f"\nFold {fold}/{N_SPLITS}")

        train_sub = [data[i] for i in train_idx]
        val_sub = [data[i] for i in val_idx]

        train_ds = SatelliteDataset(train_sub, transform=get_transforms(train=True))
        val_ds = SatelliteDataset(val_sub, transform=get_transforms(train=False))

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        model = SatelliteCNN().to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        best_acc_fold = train_one_fold(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            epochs=EPOCHS,
            name=f"fold_{fold}",
        )
        fold_accuracies.append(best_acc_fold)
        print(f"Fold {fold} Best Accuracy: {best_acc_fold:.4f}")

    avg_acc = sum(fold_accuracies) / N_SPLITS
    print("\nCross-Validation Finished!")
    print(f"Fold Accuracies: {[round(x, 4) for x in fold_accuracies]}")
    print(f"Average Accuracy: {avg_acc:.4f}")


if __name__ == "__main__":
    main()

