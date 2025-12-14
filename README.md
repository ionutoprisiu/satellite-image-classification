# Satellite Image Classification

This is a project I made to learn about deep learning and computer vision. I built a CNN using PyTorch to classify satellite images into 4 categories: water, green_area, desert, and cloudy.

## Dataset

I used a dataset with 5,631 images:
- Water: 1,500 images
- Green Area: 1,500 images  
- Desert: 1,131 images
- Cloudy: 1,500 images

All images are resized to 224x224 pixels. I split the data 80/20 for training and validation.

## Model

I built a simple CNN with 4 convolutional blocks. Each block increases the number of channels (32 -> 64 -> 128 -> 256). I used batch normalization, max pooling, and dropout to help with training.

## Experiments

I tried different things to see what works best:
- Different loss functions (CrossEntropy, WeightedCrossEntropy, KLDivLoss)
- Different optimizers (Adam, SGD, RMSprop)
- Different learning rates (0.01, 0.001, 0.0001) with different schedulers
- Different batch sizes (8, 16, 32, 64, 128)
- Cross-validation with 3 folds

## Results

The best setup I found got around 89% accuracy on validation:
- Optimizer: Adam
- Loss: CrossEntropy
- Batch size: 32
- Learning rate: 0.001

## Setup

Create a virtual environment first:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies. On Mac with Apple Silicon:

```bash
pip install torch torchvision
pip install -r requirements.txt
```

On other systems:
```bash
pip install -r requirements.txt
```

The code automatically uses MPS on Apple Silicon if available.

## Running

Make sure venv is activated, then:

```bash
python train.py
```

This will run all the experiments. It takes a while.

To see the training progress in TensorBoard:

```bash
tensorboard --logdir ~/tensorboard_logs
```

## What I learned

- How to build CNNs from scratch
- Why data augmentation is important
- How hyperparameters affect training
- How to evaluate models (confusion matrix, recall, etc.)
- Using TensorBoard to visualize training

## Things to try next

- Use pre-trained models like ResNet or EfficientNet
- Try more complex architectures
- Add more data augmentation
- Implement early stopping
- Maybe make a simple web interface

## Project Structure

```
Proiect/
├── experiments/              # experiment results go here
│   ├── class_distribution.png
│   ├── experiment_results_CrossEntropy/
│   └── ...
├── satellite-dataset/        # dataset folder
│   ├── water/
│   ├── green_area/
│   ├── desert/
│   └── cloudy/
├── train.py                  # main script
├── requirements.txt
└── README.md
```

## Notes

This was a learning project so the code has comments explaining things. All results are saved in the experiments/ folder. The code uses stratified splits to keep the class distribution balanced.
