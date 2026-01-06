
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from config import CLASS_MAPPING

def get_accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    return (preds == labels).sum().item() / labels.size(0)

def plot_confusion_matrix(labels, preds, epoch, save_dir):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    
    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=CLASS_MAPPING.keys(),
               yticklabels=CLASS_MAPPING.keys())
               
    plt.title(f'Confusion Matrix (Epoch {epoch})')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_dir, f'cm_epoch_{epoch}.png'))
    plt.close()

def plot_history(train_loss, val_loss, train_acc, val_acc, save_dir):
    # Plot loss and accuracy side by side
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train')
    plt.plot(val_loss, label='Val')
    plt.title('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Train')
    plt.plot(val_acc, label='Val')
    plt.title('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(os.path.join(save_dir, 'history.png'))
    plt.close()
