# src/data_loader.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import ssl
import os

ssl._create_default_https_context = ssl._create_unverified_context

def get_non_iid_data(uav_id, total_uavs, data_dir='data', batch_size=32):
    """
    Creates non-IID training and validation datasets for a UAV.

    Args:
        uav_id (int): The ID of the UAV.
        total_uavs (int): Total number of UAVs in the network.
        data_dir (str): Directory where the data is stored.
        batch_size (int): Batch size for data loaders.

    Returns:
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
    """
    initial_transform = transforms.Compose([transforms.ToTensor()])
    full_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=initial_transform)

    # # Calculate mean and std of the dataset
    # mean = np.mean(full_dataset.data.numpy() / 255)
    # std = np.std(full_dataset.data.numpy() / 255)

    # # Define transform with normalization
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((mean,), (std,))
    # ])

    # # Reload dataset with normalization
    # full_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    data_size_per_uav = len(full_dataset) // total_uavs
    # Get indices for this UAV
    start_idx = uav_id * data_size_per_uav
    end_idx = start_idx + data_size_per_uav
    uav_indices = list(range(start_idx, end_idx))
    # Shuffle indices to mix classes
    random.shuffle(uav_indices)
    # Create training and validation splits
    split_idx = int(len(uav_indices) * 0.8)
    train_indices = uav_indices[:split_idx]
    val_indices = uav_indices[split_idx:]
    # Create subsets
    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)
    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def get_test_data(data_dir='data', batch_size=5):
    """
    Creates a test dataset.

    Args:
        data_dir (str): Directory where the data is stored.
        batch_size (int): Batch size for the data loader.

    Returns:
        test_loader (DataLoader): DataLoader for test data.
    """
    initial_transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=initial_transform)
    
    # # Calculate mean and std of the dataset
    # mean = np.mean(test_dataset.data.numpy() / 255)
    # std = np.std(test_dataset.data.numpy() / 255)

    # # Define transform with normalization
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((mean,), (std,))
    # ])
    # test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=initial_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader