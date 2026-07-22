# Required Libraries
import numpy as np
import pandas as pd

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

import tensorflow as tf

# PyTorch DataLoader for MNIST
def get_pytorch_dataloader(batch_size=64, data_dir='./data', download=True):
    """
    Returns a PyTorch DataLoader for the MNIST dataset.
    """
    print("Loading MNIST PyTorch DataLoader...")
    
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,))
        ])
    # Load MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root=data_dir, train=True, download=download, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root=data_dir, train=False, download=download, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# TensorFlow DataLoader for MNIST
def get_tensorflow_data(batch_size=64):
    """
    Returns TensorFlow datasets for the MNIST dataset.
    """
    print("Loading MNIST dataset from Keras datasets...")

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize the images to [0, 1]
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    # Add 1 channel dimension (grayscale) to the images for CNN input (CNN expects 4 input: batch_size, height, width, channels)
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    return (x_train, y_train), (x_test, y_test)

# Scikit-Learn DataLoader for MNIST
def get_sklearn_data():
    """
    Returns Scikit-Learn datasets for the MNIST dataset.
    """
    print("Loading MNIST dataset from OpenML...")

    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist.data, mnist.target # split into features and labels

    # Preprocessing
    X = X / 255.0 # Normalize pixel values to [0, 1]

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return (X_train, y_train), (X_test, y_test)