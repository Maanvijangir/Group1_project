import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Path to raw dataset
DATA_DIR = os.path.join("data", "interim", "chest_xray")

# Define transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    #transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def get_dataloaders(batch_size=32):
    """Return train, val and test dataloaders with preprocessing applied."""
    train_data = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform)
    test_data  = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=test_transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=False)


    print("Train size:", len(train_data))
    print("Test size:", len(test_data))

    return train_loader, test_loader

""" RUN THIS FROM train.ipynb """
