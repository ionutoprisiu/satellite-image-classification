# Satellite Image Classification

This is a personal project where I built a Convolutional Neural Network (CNN) to classify satellite images into 4 different categories (water, green_area, desert, cloudy).

I wanted to learn more about Computer Vision and Deep Learning, so I implemented everything from scratch using PyTorch.

## The Model
I used a simple CNN architecture with 4 convolutional blocks. Each block has:
- A Conv2d layer
- Batch Normalization (to help with training)
- ReLU activation
- Max Pooling

The model gets around **90% accuracy** on the test set.

## Dataset
I used a dataset containing 5,631 images split into 4 classes.
- Water
- Green Area
- Desert
- Cloudy

The images are resized to 224x224 pixels. I split the data 80% for training and 20% for validation.

## How to run it

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model:
```bash
python train.py
```

The training process uses GPU acceleration automatically if available (MPS on Mac or CUDA).

## Results
During training, I save a confusion matrix for every epoch to see where the model makes mistakes.

Logs are saved in the `experiments/` folder. You can also view training progress with TensorBoard:

```bash
tensorboard --logdir runs
```

## Structure
- `config.py`: Settings like batch size and learning rate
- `model.py`: The CNN architecture code
- `dataset.py`: Code for loading and transforming images
- `train.py`: Main script to start training
