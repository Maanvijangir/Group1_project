import torch
import matplotlib.pyplot as plt
from PIL import Image
from src.dataset import test_transform
from src.explainability.gradcam import GradCAM, load_resnet50
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_resnet50("models/baseline_resnet50.pth", device)
target_layer = model.layer4[-1]  # The final convolutional layer in ResNet50
cam = GradCAM(model, target_layer)

img_path = "data/interim/chest_xray/val/PNEUMONIA/person1946_bacteria_4874.jpeg"
raw_img = Image.open(img_path).convert("RGB")
input_tensor = test_transform(raw_img).unsqueeze(0).to(device)


cams, probs, preds = cam(input_tensor)


plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original X-Ray")
plt.imshow(raw_img.resize((224, 224)))

plt.subplot(1, 2, 2)
# probs[0] is the probability of class 1 (Pneumonia)
plt.title(f"Grad-CAM (Pred: {preds[0]}, Prob: {probs[0]:.2f})")
plt.imshow(raw_img.resize((224, 224)))
plt.imshow(cams[0], cmap='jet', alpha=0.5)
import os
desktop_path = os.path.expanduser("~/Desktop/resnet50_heatmap.png")
plt.savefig("resnet50_heatmap.png")
print("Saved resnet50_heatmap.png to the project root.")