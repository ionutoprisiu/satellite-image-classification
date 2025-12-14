# Satellite image classification project
# Classifies images into: water, green_area, desert, cloudy

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Configuration
EPOCHS = 10
BATCH_SIZE = 32
NUM_CLASSES = 4
IMAGE_SIZE = (224, 224)
LEARNING_RATE = 1e-3

# Check what device to use
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

NUM_WORKERS = 0

# Dataset path
dataset_path = "satellite-dataset"

# Map class names to numbers
class_mapping = {
    'water': 0,
    'green_area': 1,
    'desert': 2,
    'cloudy': 3
}


def get_image_label_pairs(dataset_path):
    samples = []
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_path, filename)
                    samples.append((img_path, class_mapping[class_name]))
    return samples


class SatelliteDataset(Dataset):
    def __init__(self, samples, image_size=IMAGE_SIZE, transform=None):
        self.samples = samples
        self.image_size = image_size
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                  std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.image_size)
        if self.transform:
            img = self.transform(img)
        return img, label


def plot_class_distribution(samples, title="Class Distribution"):
    labels = [label for _, label in samples]
    plt.figure(figsize=(10, 6))
    plt.hist(labels, bins=NUM_CLASSES, alpha=0.7)
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(range(NUM_CLASSES), list(class_mapping.keys()), rotation=45)
    plt.tight_layout()
    
    # Save in experiments folder
    experiments_dir = 'experiments'
    os.makedirs(experiments_dir, exist_ok=True)
    output_path = os.path.join(experiments_dir, f"{title.lower().replace(' ', '_')}.png")
    plt.savefig(output_path)
    plt.close()
    return output_path


# Load dataset
print("Loading dataset...")
all_samples = get_image_label_pairs(dataset_path)
print(f"Total images: {len(all_samples)}")
class_dist_path = plot_class_distribution(all_samples)
print(f"Class distribution saved to: {class_dist_path}")

# Split into training and validation sets
train_samples, val_samples = train_test_split(
    all_samples, 
    test_size=0.2, 
    random_state=42, 
    stratify=[s[1] for s in all_samples]
)
print(f"Training samples: {len(train_samples)}")
print(f"Validation samples: {len(val_samples)}")

# Data augmentation for training
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Validation transform (no augmentation)
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets
train_dataset = SatelliteDataset(train_samples, transform=train_transform)
val_dataset = SatelliteDataset(val_samples, transform=val_transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


class SatelliteCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(SatelliteCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    return correct / total


def calculate_recall(all_preds, all_labels):
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    recalls = []
    for i in range(NUM_CLASSES):
        true_positives = ((all_preds == i) & (all_labels == i)).sum()
        actual_positives = (all_labels == i).sum()
        recall = true_positives / actual_positives if actual_positives > 0 else 0
        recalls.append(recall)
    
    return recalls


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, experiment_name=""):
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0.0
    
    experiments_dir = 'experiments'
    os.makedirs(experiments_dir, exist_ok=True)
    
    results_dir = os.path.join(experiments_dir, f'experiment_results_{experiment_name}')
    os.makedirs(results_dir, exist_ok=True)
    
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser('~'), 'tensorboard_logs', current_time)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    for epoch in range(epochs):
        print(f'\nEpoch [{epoch+1}/{epochs}]')
        
        # Training
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        total = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            acc = calculate_accuracy(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            running_acc += acc * inputs.size(0)
            total += inputs.size(0)
            
            if (batch_idx + 1) % 10 == 0:
                print(f'  Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}, Acc: {acc*100:.1f}%')
        
        epoch_train_loss = running_loss / total
        epoch_train_acc = running_acc / total
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)
        
        print(f'Train - Loss: {epoch_train_loss:.4f}, Accuracy: {epoch_train_acc*100:.1f}%')
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                acc = calculate_accuracy(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                val_acc += acc * inputs.size(0)
                total += inputs.size(0)
                
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_val_loss = val_loss / total
        epoch_val_acc = val_acc / total
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        
        recalls = calculate_recall(all_preds, all_labels)
        
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(epoch_val_loss)
            else:
                scheduler.step()
        
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=list(class_mapping.keys()),
                   yticklabels=list(class_mapping.keys()))
        plt.title(f'Confusion Matrix - Epoch {epoch+1}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        cm_path = os.path.join(results_dir, f'confusion_matrix_epoch_{epoch+1}.png')
        plt.savefig(cm_path)
        plt.close()
        
        print(f'Validation - Loss: {epoch_val_loss:.4f}, Accuracy: {epoch_val_acc*100:.1f}%')
        print('Class-wise Recall:')
        for i, class_name in enumerate(class_mapping.keys()):
            print(f'  {class_name}: {recalls[i]*100:.1f}%')
        
        writer.add_scalar(f'{experiment_name}/Loss/train', epoch_train_loss, epoch)
        writer.add_scalar(f'{experiment_name}/Loss/val', epoch_val_loss, epoch)
        writer.add_scalar(f'{experiment_name}/Accuracy/train', epoch_train_acc, epoch)
        writer.add_scalar(f'{experiment_name}/Accuracy/val', epoch_val_acc, epoch)
        
        with open(os.path.join(results_dir, 'metrics.txt'), 'a') as f:
            f.write(f'\nEpoch {epoch+1}:\n')
            f.write(f'Train Loss: {epoch_train_loss:.4f}\n')
            f.write(f'Train Accuracy: {epoch_train_acc*100:.1f}%\n')
            f.write(f'Validation Loss: {epoch_val_loss:.4f}\n')
            f.write(f'Validation Accuracy: {epoch_val_acc*100:.1f}%\n')
            f.write('Class-wise Recall:\n')
            for i, class_name in enumerate(class_mapping.keys()):
                f.write(f'  {class_name}: {recalls[i]*100:.1f}%\n')
        
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), os.path.join(results_dir, 'best_model.pth'))
            print(f'New best model saved! (Accuracy: {best_val_acc*100:.1f}%)')
        
        print('-' * 60)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='s')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy', marker='o')
    plt.plot(val_accs, label='Validation Accuracy', marker='s')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'training_curves.png'))
    plt.close()
    
    writer.close()
    
    return train_losses, val_losses, train_accs, val_accs, best_val_acc


def experiment_loss_functions():
    print("\n" + "="*60)
    print("Testing different loss functions")
    print("="*60)
    
    loss_functions = {
        'CrossEntropy': nn.CrossEntropyLoss(),
        'WeightedCrossEntropy': nn.CrossEntropyLoss(
            weight=torch.tensor([1.2, 1.0, 1.0, 1.0]).to(DEVICE)
        ),
        'KLDivLoss': nn.KLDivLoss(reduction='batchmean')
    }
    
    results = {}
    
    for loss_name, criterion in loss_functions.items():
        print(f"\nTraining with {loss_name}...")
        
        model = SatelliteCNN().to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        if loss_name == 'KLDivLoss':
            original_criterion = criterion
            def kl_criterion(outputs, targets):
                log_probs = F.log_softmax(outputs, dim=1)
                targets_one_hot = torch.zeros_like(log_probs)
                targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
                return original_criterion(log_probs, targets_one_hot)
            criterion = kl_criterion
        
        train_losses, val_losses, train_accs, val_accs, best_acc = train_model(
            model, train_loader, val_loader, criterion, optimizer, None, EPOCHS, loss_name
        )
        
        results[loss_name] = {
            'best_acc': best_acc,
            'final_train_acc': train_accs[-1],
            'final_val_acc': val_accs[-1]
        }
    
    return results


def experiment_batch_sizes():
    print("\n" + "="*60)
    print("Testing different batch sizes")
    print("="*60)
    
    batch_sizes = [8, 16, 32, 64, 128]
    results = {}
    
    # Create experiments folder if it doesn't exist
    experiments_dir = 'experiments'
    os.makedirs(experiments_dir, exist_ok=True)
    
    batch_size_dir = os.path.join(experiments_dir, 'experiment_results_batch_sizes')
    os.makedirs(batch_size_dir, exist_ok=True)
    
    for batch_size in batch_sizes:
        print(f"\nTesting batch size: {batch_size}")
        
        train_loader_bs = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
        val_loader_bs = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
        
        model = SatelliteCNN().to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()
        
        train_losses, val_losses, train_accs, val_accs, best_acc = train_model(
            model, train_loader_bs, val_loader_bs, criterion, optimizer, None, 
            epochs=5, experiment_name=f'batch_size_{batch_size}'
        )
        
        results[batch_size] = {
            'best_acc': best_acc,
            'final_train_acc': train_accs[-1],
            'final_val_acc': val_accs[-1]
        }
    
    with open(os.path.join(batch_size_dir, 'batch_size_comparison.txt'), 'w') as f:
        f.write("Batch Size Comparison\n")
        f.write("="*40 + "\n")
        for bs, result in results.items():
            f.write(f"\nBatch Size {bs}:\n")
            f.write(f"  Best Validation Accuracy: {result['best_acc']*100:.2f}%\n")
            f.write(f"  Final Training Accuracy: {result['final_train_acc']*100:.2f}%\n")
            f.write(f"  Final Validation Accuracy: {result['final_val_acc']*100:.2f}%\n")
    
    return results


def experiment_learning_rates():
    print("\n" + "="*60)
    print("Testing different learning rates and schedulers")
    print("="*60)
    
    learning_rates = [1e-2, 1e-3, 1e-4]
    schedulers = {
        'None': None,
        'StepLR': lambda opt: StepLR(opt, step_size=3, gamma=0.1),
        'CosineAnnealingLR': lambda opt: CosineAnnealingLR(opt, T_max=5)
    }
    
    for lr in learning_rates:
        for sched_name, sched_fn in schedulers.items():
            exp_name = f'LR_{lr}_{sched_name}'
            print(f"\nTesting: Learning Rate={lr}, Scheduler={sched_name}")
            
            model = SatelliteCNN().to(DEVICE)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            scheduler = sched_fn(optimizer) if sched_fn else None
            criterion = nn.CrossEntropyLoss()
            
            train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, EPOCHS, exp_name)
    
    print("\nFinished learning rate experiments.")
    return True


def experiment_optimizers():
    print("\n" + "="*60)
    print("Testing different optimizers")
    print("="*60)
    
    optimizers_cfg = {
        'Adam': lambda params: optim.Adam(params, lr=LEARNING_RATE),
        'SGD': lambda params: optim.SGD(params, lr=LEARNING_RATE, momentum=0.9),
        'RMSprop': lambda params: optim.RMSprop(params, lr=LEARNING_RATE)
    }
    
    loss_functions = {
        'CrossEntropy': nn.CrossEntropyLoss(),
        'WeightedCrossEntropy': nn.CrossEntropyLoss(
            weight=torch.tensor([1.2, 1.0, 1.0, 1.0]).to(DEVICE)
        )
    }
    
    results = {}
    
    for opt_name, opt_fn in optimizers_cfg.items():
        for loss_name, criterion in loss_functions.items():
            exp_name = f'{opt_name}_{loss_name}'
            print(f"\nTesting: Optimizer={opt_name}, Loss={loss_name}")
            
            model = SatelliteCNN().to(DEVICE)
            optimizer = opt_fn(model.parameters())
            
            train_model(model, train_loader, val_loader, criterion, optimizer, None, EPOCHS, exp_name)
            
            results[exp_name] = {'model': model}
    
    return results


def cross_validation(k_folds=3):
    print("\n" + "="*60)
    print(f"Running {k_folds}-fold cross-validation")
    print("="*60)
    
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(all_samples)):
        print(f"\nFold {fold + 1}/{k_folds}")
        
        train_samples_fold = [all_samples[i] for i in train_idx]
        val_samples_fold = [all_samples[i] for i in val_idx]
        
        train_dataset_fold = SatelliteDataset(train_samples_fold, transform=train_transform)
        val_dataset_fold = SatelliteDataset(val_samples_fold, transform=val_transform)
        
        train_loader_fold = DataLoader(train_dataset_fold, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        val_loader_fold = DataLoader(val_dataset_fold, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        
        model = SatelliteCNN().to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
        
        train_losses, val_losses, train_accs, val_accs, best_acc = train_model(
            model, train_loader_fold, val_loader_fold, criterion, optimizer, scheduler, EPOCHS, f'CV_fold_{fold+1}'
        )
        
        fold_results.append({
            'fold': fold + 1,
            'best_acc': best_acc,
            'final_val_acc': val_accs[-1]
        })
    
    print("\nCross-Validation Results:")
    print("-" * 40)
    for result in fold_results:
        print(f"Fold {result['fold']}: Best Accuracy = {result['best_acc']*100:.2f}%")
    avg_acc = np.mean([r['best_acc'] for r in fold_results])
    print(f"\nAverage Accuracy: {avg_acc*100:.2f}%")
    
    return fold_results


if __name__ == "__main__":
    print("="*60)
    print("Satellite Image Classification Project")
    print("="*60)
    device_str = str(DEVICE)
    if device_str == "mps":
        device_str = "MPS (Apple Silicon GPU)"
    elif device_str.startswith("cuda"):
        device_str = f"CUDA ({DEVICE})"
    else:
        device_str = "CPU"
    print(f"Device: {device_str}")
    print(f"Number of classes: {NUM_CLASSES}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print("="*60)
    
    print("\nStarting experiments...")
    
    print("\n1. Loss functions")
    loss_results = experiment_loss_functions()
    
    print("\n2. Batch sizes")
    batch_results = experiment_batch_sizes()
    
    print("\n3. Learning rates")
    lr_results = experiment_learning_rates()
    
    print("\n4. Optimizers")
    opt_results = experiment_optimizers()
    
    print("\n5. Cross-validation")
    cv_results = cross_validation()
    
    print("\n" + "="*60)
    print("All experiments completed!")
    print("="*60)
    print("\nResults saved in experiments/ folder")
    print("Check TensorBoard for visualizations:")
    print(f"  tensorboard --logdir ~/tensorboard_logs")

