# Computer Vision Project - Pneumonia Detection Using CNN

## Overview

This project focuses on applying **Convolutional Neural Networks (CNNs)** for chest X-ray image classification and pneumonia detection. The workflow is designed to be reproducible, lightweight, and team-friendly, enabling easy dataset setup, preprocessing, training, and evaluation.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Dataset

- **Source:** Kaggle - Chest X-Ray Pneumonia Dataset (Paul Mooney)
- Downloaded via **Kaggle API** in `setup.ipynb`
- Zipped dataset stored in `data/raw/`
- Extracted dataset stored in `data/interim/`
- Preprocessing handled by `src/dataset.py` (transforms, normalization, and dataloaders)

---

## Repository Structure

```text
├── data/
│   ├── raw/          # Placeholder for downloaded zip dataset
│   ├── interim/      # Extracted original dataset
│   └── processed/    # Placeholder for processed data
│
├── reports/
│   ├── EDA_Report.md
│   ├── EDA_Summary.csv
│   ├── explainability_summary.csv
│   ├── image_aspect_ratio_summary.csv
│   ├── image_metadata.csv
│   └── ...
│
├── src/
│   ├── dataset.py    # Preprocessing and dataloader implementation
│   ├── modeling/train.py  # ResNet-50 baseline training
│   └── explainability/
│       ├── gradcam.py    # Grad-CAM, occlusion sensitivity, border-attention metric
│       └── test_cam.py   # Quick Grad-CAM check for the ResNet-50 on one X-ray
|
├── EDA.ipynb         # Exploratory Data Analysis
├── setup.ipynb       # Kaggle API integration and dataset extraction
├── train_optimized.py # ResNet-18 training script
├── evaluate.py       # Accuracy, recall, specificity, precision, F1, ROC-AUC, confusion matrix
├── explain.py        # Explainability script (Grad-CAM + occlusion)
├── explainability.ipynb # Explainability walkthrough and findings
├── requirements.txt
└── README.md
```

---

## Exploratory Data Analysis (EDA)

Data exploration was performed in `EDA.ipynb` to better understand the dataset before training.

The analysis includes:

- Class distribution analysis
- Image size and resolution statistics
- Aspect ratio analysis
- Image metadata generation
- Dataset quality checks
- Sample image visualization

Generated reports are available in the `reports/` directory.

### Available Reports

| File | Description |
|--------|-------------|
| `EDA_Report.md` | Written EDA report |
| `EDA_Summary.csv` | Summary statistics and EDA findings |
| `image_aspect_ratio_summary.csv` | Aspect ratio distribution analysis |
| `image_metadata.csv` | Extracted image metadata |
| Other report files | Additional dataset analysis outputs |

---

## Model Training

`train_optimized.py` trains the ResNet-18:

- Loads an 85/15 stratified train/validation split from `src/dataset.py`
- Balances the classes with `pos_weight` (normal images / pneumonia images)
- Trains the final layer for 3 epochs, then the whole network, for up to 15 epochs with early stopping
- Chooses the threshold on the validation set each epoch and keeps the best epoch
- Prints the final test-set results
- Saves the weights to `pneumonia_resnet18_recall_optimized.pth` and the log to `performance_logs_optimization.csv`

`src/modeling/train.py` trains the ResNet-50 baseline and saves it to `models/baseline_resnet50.pth`, which is not stored in this repo.

---

## Results

### Model Performance

The current model is the ResNet-18 in `pneumonia_resnet18_recall_optimized.pth`, trained by `train_optimized.py` with the 224×224 center crop. ResNet-50 weights are not in this repo, so it could not be compared.

Results on the 624 test images:

| Metric | Threshold 0.52 (from training) | Threshold 0.88 (chosen on validation for ≥95% recall) |
|---------|---------|---------|
| Recall | 99.5% | 98.5% |
| Specificity | 69.7% | 81.2% |
| Precision | 84.5% | 89.7% |
| Accuracy | 88.3% | 92.0% |
| F1 | 0.914 | 0.939 |
| Missed pneumonia / false alarms | 2 / 71 | 6 / 44 |

ROC-AUC is 0.975 for both, since it does not depend on the threshold.

### Key Findings

- The model finds **99.5% of pneumonia cases**, but at the training threshold it still flags **71 of 234 normal X-rays** as pneumonia.
- Raising the threshold to 0.88 cuts false alarms to 44 while recall stays at 98.5%. The team still needs to choose the threshold.
- The previous ResNet-18 (150×150, `pos_weight=3`) missed 12 cases and flagged 133 normal X-rays.

---

## Evaluation

`evaluate.py` reports accuracy, recall, specificity, precision, F1, ROC-AUC and the confusion matrix on the validation and test splits, and saves them to `reports/evaluation_<arch>.csv` with a confusion matrix and ROC curve in `reports/figures/evaluation/`.

```bash
python evaluate.py                                # ResNet-18, threshold from performance_logs_optimization.csv
python evaluate.py --min-recall 0.95              # pick the threshold on validation, apply it to test
python evaluate.py --arch resnet50                # ResNet-50 from models/baseline_resnet50.pth
python evaluate.py --arch resnet50 --min-recall 0.95   # pick the threshold on validation, apply it to test
```

Always quote **test** results. Validation scores are optimistic: the ResNet-18 scores 99.0% specificity on validation but 69.7% on test. Most pneumonia validation images share a patient with a training image.

Without `--threshold`, the ResNet-18 uses the threshold `train_optimized.py` chose for its best epoch, read from `performance_logs_optimization.csv`.

---

## Explainability

High recall does not show *why* the model predicts pneumonia. Chest X-ray models are known to pick up shortcuts such as text markers, hospital labels or image borders instead of lung tissue, so the models are checked with two methods:

- **Grad-CAM** on the last convolutional block (`layer4`) highlights the regions that drove each prediction.
- **Occlusion sensitivity** hides one 15×15 patch at a time and measures the drop in pneumonia probability. It does not use gradients, so it cross-checks Grad-CAM.

Run from the repo root once the dataset is in `data/interim/chest_xray`:

```bash
python explain.py                                          # ResNet-18, whole test split
python explain.py --image path/to/xray.jpeg                # ResNet-18, a single X-ray
python explain.py --arch resnet50                          # ResNet-50 from models/baseline_resnet50.pth
```

Outputs:

| File | Description |
|--------|-------------|
| `reports/figures/explainability/<run>/<outcome>.png` | Grad-CAM examples for true/false positives and negatives |
| `reports/figures/explainability/<run>/occlusion_<outcome>.png` | Grad-CAM next to occlusion for the same images |
| `reports/explainability_summary_<run>.csv` | Per-image probability, prediction, outcome and `cam_border_fraction` |

`<run>` is the model name plus any `--image-size`, for example `resnet18`, so one model's report does not overwrite another's. The saved report for the current ResNet-18 is in `reports/explainability_summary.csv` and `reports/figures/explainability/`.

`cam_border_fraction` is the share of the heatmap in the outer 15% of the image. A flat heatmap scores about 0.51; values well above that point to border artefacts rather than the lungs. `explainability.ipynb` walks through the figures and records the findings.

Use `--image-size` only for older weights trained at another size without the center crop, such as the previous 150×150 ResNet-18.

### Explainability Findings (current ResNet-18, test set, threshold 0.52)

| Outcome | Count | Mean border attention | Previous model |
|---------|-------|-----------------------|----------------|
| True positive | 388 | 0.43 | 378, 0.49 |
| True negative | 163 | 0.42 | 101, 0.30 |
| False positive | 71 | 0.47 | 133, 0.59 |
| False negative | 2 | 0.43 | 12, 0.32 |

- The model now **looks mostly at the lungs**: only 12.3% of heatmaps have more border attention than a flat heatmap, down from 40.9%.
- **False alarms remain the weakness.** In the occlusion examples, image edges and the area below the diaphragm push healthy X-rays towards pneumonia.
- Both missed cases focus on the trachea and upper mediastinum, the same pattern as the previous model's 12 misses.
- Details and figures are in `explainability.ipynb`.

---

## Notes

- Datasets are **not pushed to GitHub**; only the folder structure is included.
- Run `setup.ipynb` to download the dataset, then the scripts above to train and evaluate. ResNet-50 weights are not in this repo.
- Users must download the dataset locally and place it in the appropriate `data/` directories.
- Generated reports can be found in the `reports/` folder.

