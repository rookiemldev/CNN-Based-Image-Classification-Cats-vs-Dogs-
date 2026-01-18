# Cats vs Dogs Image Classification (CNN)

This project implements a Convolutional Neural Network (CNN) using Keras to perform binary image classification (Cats vs Dogs).

## Overview
- Input: RGB images resized to 100×100
- Output: Binary classification (Cat / Dog)
- Model: Custom CNN trained from scratch
- Framework: Keras (TensorFlow backend)

## Model Architecture
- Conv2D (32 filters, 3×3) + ReLU
- MaxPooling2D (2×2)
- Conv2D (32 filters, 3×3) + ReLU
- MaxPooling2D (2×2)
- Fully connected layer (64 units)
- Sigmoid output for binary classification

## Training Details
- Loss: Binary Crossentropy
- Optimizer: Adam
- Epochs: 10
- Batch size: 64
- Normalization: Pixel values scaled to [0,1]

## Results
The model achieves competitive accuracy on the test set, demonstrating effective feature extraction using CNNs.

