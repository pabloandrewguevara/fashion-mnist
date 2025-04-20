# FashionMNIST CNN Classifier

This repository contains a Convolutional Neural Network (CNN) implementation using PyTorch to classify images from the FashionMNIST dataset.

## Dataset

The FashionMNIST dataset consists of 28x28 grayscale images of fashion items from 10 classes:

- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

## Model Architecture

- **Convolutional Layers**: Extract features from images using two convolutional layers.
- **Pooling Layers**: Reduce spatial dimensions using max pooling.
- **Fully Connected Layers**: Classify the flattened feature maps into 10 classes.
- **Dropout**: Included to mitigate overfitting.

