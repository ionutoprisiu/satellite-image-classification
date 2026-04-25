# Satellite Image Classification

A PyTorch project for classifying satellite images into 4 land-cover categories:

- `water`
- `green_area`
- `desert`
- `cloudy`

The project uses a custom CNN and trains/evaluates it with **5-Fold Cross Validation**.

## Features

- Custom CNN with 4 convolutional blocks (`Conv2d + BatchNorm + ReLU + MaxPool`)
- Data augmentation for training
- Automatic device selection (`MPS` on Apple Silicon, then `CUDA`, then `CPU`)
- 5-fold training pipeline with per-fold best checkpoint saving
- Stratified fold splitting for more balanced validation sets
- TensorBoard logging
- Confusion matrix saved at every epoch
- Training history plots (loss and accuracy)

## Tech Stack

- Python
- PyTorch / torchvision
- OpenCV
- scikit-learn
- TensorBoard

## Project Structure

- `src/satellite_image_classification/` - core ML package
- `scripts/train.py` - training entrypoint
- `tests/` - lightweight smoke tests
- `artifacts/` - generated outputs (created automatically)
- `satellite-dataset/` - local dataset directory

## Dataset Format

Place your dataset in a folder named `satellite-dataset` at the project root:

```text
satellite-dataset/
  water/
  green_area/
  desert/
  cloudy/
```

Each class folder should contain image files with extensions:

- `.jpg`
- `.jpeg`
- `.png`

Images are resized to `224x224` during loading.

## Installation

```bash
pip install -r requirements.txt
```

Optional (recommended for a clean project workflow):

```bash
pip install -e ".[dev]"
```

## Training

```bash
python scripts/train.py
```

After editable install, you can also run:

```bash
sat-train
```

Default settings are defined in `src/satellite_image_classification/config.py`:

- `EPOCHS = 10`
- `BATCH_SIZE = 32`
- `LEARNING_RATE = 1e-3`
- `IMAGE_SIZE = (224, 224)`
- `NUM_WORKERS = 0`
- `SEED = 42`

## Outputs

For each fold, the training script creates:

- `artifacts/experiments/fold_X/best_model.pth` (best checkpoint by validation accuracy)
- confusion matrix images per epoch
- loss/accuracy history plots

TensorBoard logs are written to `artifacts/runs/`:

```bash
tensorboard --logdir artifacts/runs
```

## Testing

```bash
pytest
```

## Known Limitations

- No dedicated holdout test set in the current workflow
- Checkpoints save model weights only (no optimizer/epoch state)

## Notes

- The pipeline uses **K-Fold validation**, not a fixed 80/20 split
- Reported performance depends on the dataset and fold split

## What I Learned

- How to build an end-to-end AI pipeline in PyTorch (data loading, model, training, evaluation)
- Why validation strategy matters, and why cross-validation gives more reliable results than a single split
- How data augmentation helps model generalization
- Why accuracy alone is not enough, and when to look at precision, recall, and F1
- How to debug model behavior using confusion matrices and TensorBoard logs
