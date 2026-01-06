
import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from config import IMAGE_SIZE, CLASS_MAPPING

def get_image_label_pairs(dataset_path):
    # Walk through folders and find images
    data = []
    for class_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(folder_path):
            continue
            
        if class_name in CLASS_MAPPING:
            label = CLASS_MAPPING[class_name]
            for file in os.listdir(folder_path):
                if file.endswith(('.jpg', '.png', '.jpeg')):
                    data.append((os.path.join(folder_path, file), label))
    return data

def get_transforms(train=True):
    # Add augmentation for training data
    if train:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    # Just normalize for validation
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

class SatelliteDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform or get_transforms(train=False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        
        # Read and process image
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Bad image: {path}")
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMAGE_SIZE)
        
        if self.transform:
            img = self.transform(img)
            
        return img, label
