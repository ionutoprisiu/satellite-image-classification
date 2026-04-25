import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from .config import CLASS_MAPPING

CLASS_LABELS = list(range(len(CLASS_MAPPING)))
CLASS_NAMES = [name for name, _ in sorted(CLASS_MAPPING.items(), key=lambda item: item[1])]


def get_accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    return (preds == labels).sum().item() / labels.size(0)


def get_classification_metrics(labels, preds):
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        labels=CLASS_LABELS,
        average="macro",
        zero_division=0,
    )
    return precision, recall, f1


def plot_confusion_matrix(labels, preds, epoch, save_dir):
    cm = confusion_matrix(labels, preds, labels=CLASS_LABELS)
    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )

    plt.title(f"Confusion Matrix (Epoch {epoch})")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()

    plt.savefig(os.path.join(save_dir, f"cm_epoch_{epoch}.png"))
    plt.close()


def plot_history(train_loss, val_loss, train_acc, val_acc, save_dir):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label="Train")
    plt.plot(val_loss, label="Val")
    plt.title("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label="Train")
    plt.plot(val_acc, label="Val")
    plt.title("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(save_dir, "history.png"))
    plt.close()

