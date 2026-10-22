import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Path to raw dataset
RAW_DIR = os.path.join("data", "raw", "chest_xray")

# Define transforms
train_transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def get_dataloaders(batch_size=32):
    """Return train and test dataloaders with preprocessing applied."""
    train_data = datasets.ImageFolder(os.path.join(RAW_DIR, "train"), transform=train_transform)
    test_data  = datasets.ImageFolder(os.path.join(RAW_DIR, "test"), transform=test_transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    print("Train size:", len(train_data))
    print("Test size:", len(test_data))

    return train_loader, test_loader

""" TO RUN THIS IN NOTEBOOK 
from src.dataset import get_dataloaders

train_loader, test_loader = get_dataloaders(batch_size=32) """