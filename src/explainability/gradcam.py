import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models

# ImageFolder sorts class folders alphabetically: NORMAL=0, PNEUMONIA=1
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


def load_resnet18(weights_path, device):
    """Rebuild the ResNet-18 from train_optimized.py (single logit) and load its weights."""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    return model.to(device).eval()


def load_resnet50(weights_path, device):
    """Rebuild the ResNet-50 from src/modeling/train.py (two logits) and load its weights."""
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    return model.to(device).eval()


LOADERS = {"resnet18": load_resnet18, "resnet50": load_resnet50}


def pneumonia_probability(logits):
    """P(pneumonia) for either head: one sigmoid logit (ResNet-18) or two softmax logits (ResNet-50)."""
    if logits.shape[1] == 1:
        return torch.sigmoid(logits.view(-1))
    return F.softmax(logits, dim=1)[:, 1]


class GradCAM:
    """Grad-CAM (Selvaraju et al., 2017) for the ResNet-18 (one logit) or ResNet-50 (two logits).

    The heatmap explains the predicted class. With one logit that is the pneumonia
    logit for a PNEUMONIA prediction and the negated logit for NORMAL; with two
    logits it is the logit of the predicted class.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self._handle = target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, inputs, output):
        self.activations = output
        # Forward passes under no_grad (e.g. occlusion) have nothing to hook
        if output.requires_grad:
            output.register_hook(self._save_gradient)

    def _save_gradient(self, grad):
        self.gradients = grad

    def remove(self):
        self._handle.remove()

    def __call__(self, images, threshold=0.5):
        """Return (cams, probabilities, predictions) for a batch of normalized images.

        probabilities are P(pneumonia) with shape (N,); cams has shape (N, H, W)
        at input resolution, each scaled to [0, 1].
        """
        self.model.zero_grad()
        with torch.enable_grad():
            logits = self.model(images)
            probs = pneumonia_probability(logits)
            preds = (probs >= threshold).long()
            # Samples are independent in eval mode, so one backward pass on the sum
            # gives every sample's gradient at once
            if logits.shape[1] == 1:
                sign = torch.where(preds == 1, 1.0, -1.0)
                score = (logits.view(-1) * sign).sum()
            else:
                score = logits.gather(1, preds.view(-1, 1)).sum()
            score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cams = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cams = F.interpolate(cams, size=images.shape[-2:], mode="bilinear", align_corners=False)
        cams = cams.squeeze(1)

        flat = cams.view(cams.size(0), -1)
        low, high = flat.min(dim=1)[0], flat.max(dim=1)[0]
        cams = (cams - low[:, None, None]) / (high - low + 1e-8)[:, None, None]
        return cams.detach().cpu().numpy(), probs.detach().cpu().numpy(), preds.cpu().numpy()


def occlusion_sensitivity(model, image, patch=15, stride=15, batch_size=64):
    """Slide a gray patch over one normalized image (C, H, W) and record how much the
    pneumonia probability drops. Used to cross-check Grad-CAM with a model-agnostic method.
    """
    device = image.device
    _, height, width = image.shape
    with torch.no_grad():
        base = pneumonia_probability(model(image.unsqueeze(0)))[0].item()

    positions = [(y, x) for y in range(0, height - patch + 1, stride)
                 for x in range(0, width - patch + 1, stride)]
    drops = []
    for start in range(0, len(positions), batch_size):
        chunk = positions[start:start + batch_size]
        batch = image.unsqueeze(0).repeat(len(chunk), 1, 1, 1)
        for i, (y, x) in enumerate(chunk):
            # 0 in normalized space is mid-gray for Normalize((0.5,), (0.5,))
            batch[i, :, y:y + patch, x:x + patch] = 0.0
        with torch.no_grad():
            probs = pneumonia_probability(model(batch))
        drops.extend((base - probs).cpu().tolist())

    heatmap = np.zeros((height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32)
    for (y, x), drop in zip(positions, drops):
        heatmap[y:y + patch, x:x + patch] += drop
        counts[y:y + patch, x:x + patch] += 1
    return heatmap / np.maximum(counts, 1)


def border_fraction(cam, band=0.15):
    """Share of the heatmap's mass lying in the outer band of the image.

    Lungs sit in the middle of a chest X-ray, while text markers, labels and
    scanner borders sit near the edges. A high value suggests the model may rely
    on these shortcuts instead of lung tissue.
    """
    height, width = cam.shape
    dy, dx = int(height * band), int(width * band)
    total = cam.sum() + 1e-8
    inner = cam[dy:height - dy, dx:width - dx].sum()
    return float((total - inner) / total)


def denormalize(image):
    """Undo Normalize((0.5,), (0.5,)) and return an (H, W) grayscale array in [0, 1]."""
    return (image[0] * 0.5 + 0.5).clamp(0, 1).cpu().numpy()
