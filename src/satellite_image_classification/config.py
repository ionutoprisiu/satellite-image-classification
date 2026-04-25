from pathlib import Path
import torch

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "satellite-dataset"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
EXPERIMENTS_DIR = ARTIFACTS_DIR / "experiments"
RUNS_DIR = ARTIFACTS_DIR / "runs"

# Training settings
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
IMAGE_SIZE = (224, 224)
NUM_WORKERS = 0
SEED = 42
N_SPLITS = 5

# Class labels
CLASS_MAPPING = {
    "water": 0,
    "green_area": 1,
    "desert": 2,
    "cloudy": 3,
}

NUM_CLASSES = len(CLASS_MAPPING)

# Select device (MPS for Mac, CUDA for NVIDIA, or CPU)
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

