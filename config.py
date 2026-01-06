
import torch

# Training settings
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
IMAGE_SIZE = (224, 224)
NUM_WORKERS = 0

# Class labels
CLASS_MAPPING = {
    'water': 0,
    'green_area': 1,
    'desert': 2,
    'cloudy': 3
}

NUM_CLASSES = len(CLASS_MAPPING)

# Select device (MPS for Mac, CUDA for NVIDIA, or CPU)
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
