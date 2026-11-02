import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader
from sklearn.model_selection import train_test_split

# Path to raw dataset
DATA_DIR = os.path.join("data", "interim", "chest_xray")

# Share of the training images held out for validation, and the seed that makes
# the split identical on every machine
VAL_FRACTION = 0.2
SPLIT_SEED = 42

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


class XrayDataset(Dataset):
    """A list of (path, label) samples with its own transform."""

    def __init__(self, samples, transform):
        self.samples = samples
        self.targets = [label for _, label in samples]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        return self.transform(default_loader(path)), label


def get_train_val_samples(val_fraction=VAL_FRACTION, seed=SPLIT_SEED):
    """Split the training images into stratified train and validation samples.

    The official val/ folder has only 16 images, so it is pooled with train/ when
    it still exists (setup.ipynb moves it into train/). Samples are sorted by class
    and file name first, so both folder layouts give exactly the same split.
    """
    samples = datasets.ImageFolder(os.path.join(DATA_DIR, "train")).samples
    val_dir = os.path.join(DATA_DIR, "val")
    if os.path.isdir(val_dir):
        samples += datasets.ImageFolder(val_dir).samples
    samples = sorted(samples, key=lambda s: (s[1], os.path.basename(s[0])))

    labels = [label for _, label in samples]
    train_samples, val_samples = train_test_split(
        samples, test_size=val_fraction, stratify=labels, random_state=seed)
    return train_samples, val_samples


def get_dataloaders(batch_size=32):
    """Return train, val and test dataloaders with preprocessing applied."""
    train_samples, val_samples = get_train_val_samples()
    train_data = XrayDataset(train_samples, train_transform)
    val_data = XrayDataset(val_samples, test_transform)
    test_data  = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=test_transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader  = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=False)


    print("Train size:", len(train_data))
    print("Valid size:", len(val_data))
    print("Test size:", len(test_data))

    return train_loader, val_loader, test_loader

""" RUN THIS FROM train.ipynb """
