
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from datetime import datetime

# Local imports
from config import *
from dataset import SatelliteDataset, get_image_label_pairs, get_transforms
from model import SatelliteCNN
from utils import get_accuracy, plot_confusion_matrix, plot_history

def train(model, train_loader, val_loader, criterion, optimizer, epochs, name="exp"):
    print(f"\nStarting {name}...")
    
    # Logging
    log_dir = os.path.join("runs", f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{name}")
    writer = SummaryWriter(log_dir)
    results_dir = os.path.join("experiments", name)
    os.makedirs(results_dir, exist_ok=True)
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_acc = 0.0
    
    for epoch in range(epochs):
        # Training loop
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
        
        # Validation loop
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
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f} | Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(results_dir, "best_model.pth"))
            
        # Save confusion matrix every epoch
        plot_confusion_matrix(all_labels, all_preds, epoch+1, results_dir)

    # Plot final history
    plot_history(train_losses, val_losses, train_accs, val_accs, results_dir)
    writer.close()
    return best_acc

def main():
    print(f"Using device: {DEVICE}")
    
    # 1. Prepare Data
    print("Loading data...")
    data = get_image_label_pairs("satellite-dataset")
    train_data, val_data = train_test_split(data, test_size=0.2, shuffle=True, stratify=[x[1] for x in data])
    
    train_ds = SatelliteDataset(train_data, transform=get_transforms(train=True))
    val_ds = SatelliteDataset(val_data, transform=get_transforms(train=False))
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    # 2. Setup Model
    model = SatelliteCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 3. Train
    print("Starting Main Training...")
    best_acc = train(model, train_loader, val_loader, criterion, optimizer, EPOCHS, name="baseline")
    print(f"Training finished! Best accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    main()
