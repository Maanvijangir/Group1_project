"""Quick Grad-CAM check for the ResNet-50 on one test X-ray.

Usage (from the repo root):
  python src/explainability/test_cam.py [path/to/xray.jpeg]
"""
import os
import sys

# Make "src" importable when run as a script from the repo root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from src.dataset import DATA_DIR, test_transform
from src.explainability.gradcam import CLASS_NAMES, GradCAM, load_resnet50


device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model = load_resnet50("models/baseline_resnet50.pth", device)
target_layer = model.layer4[-1]  # The final convolutional layer in ResNet50
cam = GradCAM(model, target_layer)

# Use a test image: the old val/ images are now part of the training data
img_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    DATA_DIR, "test", "PNEUMONIA", "person100_bacteria_475.jpeg")
raw_img = Image.open(img_path).convert("RGB")
input_tensor = test_transform(raw_img).unsqueeze(0).to(device)


cams, probs, preds = cam(input_tensor)
cam.remove()

# The heatmap has the model's input size, so show the image at that size too
size = tuple(input_tensor.shape[-2:][::-1])
display_img = raw_img.resize(size)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original X-Ray")
plt.imshow(display_img)
plt.axis("off")

plt.subplot(1, 2, 2)
# probs[0] is the probability of class 1 (Pneumonia)
plt.title(f"Grad-CAM (Pred: {CLASS_NAMES[preds[0]]}, P(pneumonia): {probs[0]:.2f})")
plt.imshow(display_img)
plt.imshow(cams[0], cmap='jet', alpha=0.5)
plt.axis("off")

out_dir = os.path.join("reports", "figures", "explainability")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "resnet50_test_cam.png")
plt.savefig(out_path, bbox_inches="tight")
print(f"Saved {out_path}")
