# Multi-Class Image Classification Using CNN on CIFAR-10 Dataset

This project implements a multi-class image classification model using a Convolutional Neural Network (CNN) built with Keras and TensorFlow to classify images from the CIFAR-10 dataset.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Dataset Classes](#dataset-classes)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Overview

The goal of this project is to classify images into one of ten classes in the CIFAR-10 dataset. The model processes images and predicts the class based on its features.

## Dataset

The dataset used in this project is the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html), which consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class.

### Dataset Classes

The CIFAR-10 dataset contains the following classes:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

## Model Architecture

The CNN architecture for the CIFAR-10 classification is designed with several convolutional and pooling layers to extract features from the input images. It begins with an input layer that takes images of size 32x32 pixels with three color channels (RGB). 

- The model consists of three convolutional layers, each followed by a max-pooling layer, which reduces the spatial dimensions and helps to retain the most significant features.
- A batch normalization layer is included after the third convolutional layer to stabilize and accelerate training.
- The model is flattened into a one-dimensional vector before passing it through a dense layer with ReLU activation, leading to an output layer that utilizes softmax activation to classify the images into one of the ten categories.

## Installation

To run this project, you need to have Python installed along with the following libraries:

- TensorFlow
- Keras
- NumPy
- Matplotlib

You can install the required libraries using pip:

```bash
pip install tensorflow keras numpy matplotlib
