import os
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader
from sklearn.model_selection import train_test_split

# Path to raw dataset (kept identical for explain.py / gradcam.py compatibility)
DATA_DIR = os.path.join("data", "interim", "chest_xray")

# Training transforms: crop out outer ~15% to remove "A-P" / "R" markers & scanner borders
# Share of the training images held out for validation, and the seed that makes
# the split identical on every machine
VAL_FRACTION = 0.2
SPLIT_SEED = 42

# Define transforms
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomRotation(5),
    transforms.CenterCrop(224),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Validation / Test transforms: deterministic center crop to 224x224
test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


def get_test_transform(image_size=None):
    """test_transform, or plain resizing for a model trained at another size without the crop."""
    if image_size is None:
        return test_transform
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])


class SubsetWithTransform(Dataset):
    """Applies split-specific transforms without leaking augmentations to validation."""
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        if self.transform:
            img = self.transform(img)
        return img, label


def _extract_stratification_key(filepath, label):
    """Stratifies across Normal (0), Bacterial Pneumonia (1), and Viral Pneumonia (2)."""
    if label == 0:
        return 0
    fname = os.path.basename(filepath).lower()
    if "virus" in fname:
        return 2
    return 1


def get_dataloaders(batch_size=32, val_split=0.15, return_val=True, num_workers=2):
    """
    Returns DataLoaders with 85/15 stratified train/val split:
    (train_loader, val_loader, test_loader), or (train_loader, test_loader) with return_val=False.
    """
    train_dir = os.path.join(DATA_DIR, "train")
    val_dir = os.path.join(DATA_DIR, "val")
    test_dir = os.path.join(DATA_DIR, "test")

    # Load raw PIL datasets without transforms first
    base_train = datasets.ImageFolder(train_dir, transform=None)
    datasets_list = [base_train]
    all_samples = list(base_train.samples)

    # setup.ipynb moves the val/ images into train/ and leaves the folders empty
    try:
        base_val = datasets.ImageFolder(val_dir, transform=None)
    except FileNotFoundError:
        base_val = None
    if base_val is not None:
        datasets_list.append(base_val)
        all_samples.extend(base_val.samples)

    full_pool = ConcatDataset(datasets_list)
    strat_keys = [_extract_stratification_key(p, lbl) for p, lbl in all_samples]
    indices = list(range(len(full_pool)))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_split,
        random_state=42,
        stratify=strat_keys
    )

    train_data = SubsetWithTransform(full_pool, train_idx, transform=train_transform)
    val_data = SubsetWithTransform(full_pool, val_idx, transform=test_transform)
    test_data = datasets.ImageFolder(test_dir, transform=test_transform)

    # Store targets for class weight calculation
    train_data.targets = [all_samples[i][1] for i in train_idx]
    val_data.targets = [all_samples[i][1] for i in val_idx]

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)

    print(f"Train size: {len(train_data)}")
    print(f"Val size:   {len(val_data)}")
    print(f"Test size:  {len(test_data)}")

    if return_val:
        return train_loader, val_loader, test_loader
    return train_loader, test_loader