import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from src.dataset import get_dataloaders

def train_and_evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=32)

    positive_weight = torch.tensor([3.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device).float()
        optimizer.zero_grad()
        loss = criterion(model(images).squeeze(), labels)
        loss.backward()
        optimizer.step()

    model.eval()
    test_true_positives, test_actual_positives = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device).float()
            probabilities = torch.sigmoid(model(images).squeeze())
            final_guesses = (probabilities >= 0.4).float()
            
            for guess, actual in zip(final_guesses, labels):
                if actual == 1:
                    test_actual_positives += 1
                    if guess == 1: test_true_positives += 1

    torch.save(model.state_dict(), 'pneumonia_resnet18_recall_optimized.pth')

if __name__ == "__main__":
    train_and_evaluate()
