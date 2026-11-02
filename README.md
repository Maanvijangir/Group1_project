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

The weights currently in this repo were trained with an earlier version of `train_optimized.py`, at 150×150. Their results on the 624 test images, with threshold 0.4:

| Metric | Score |
|---------|---------|
| Recall | 96.9% |
| Specificity | 43.2% |
| Precision | 74.0% |
| Accuracy | 76.8% |
| F1 | 0.839 |
| ROC-AUC | 0.913 |

Confusion matrix: 378 true positives, 12 false negatives, 101 true negatives, 133 false positives.

### Key Findings

- The model finds **96.9% of pneumonia cases**, but it also flags **133 of 234 normal X-rays** as pneumonia, so these results are not ready for deployment.
- The earlier **100% validation recall** came from the original validation folder, which holds only 8 pneumonia images, so it is not a reliable estimate.
- The model prioritizes high recall, which is critical in medical diagnosis tasks where missing pneumonia cases can have serious consequences.

---

## Evaluation

`evaluate.py` reports accuracy, recall, specificity, precision, F1, ROC-AUC and the confusion matrix on the validation and test splits, and saves them to `reports/evaluation_<arch>.csv` with a confusion matrix and ROC curve in `reports/figures/evaluation/`.

```bash
python evaluate.py --image-size 150               # ResNet-18 weights in this repo (trained at 150x150), threshold 0.4
python evaluate.py --arch resnet50                # ResNet-50 from models/baseline_resnet50.pth
python evaluate.py --arch resnet50 --min-recall 0.95   # pick the threshold on validation, apply it to test
```

Always quote **test** results. Validation scores can be optimistic: the ResNet-18 in this repo scores 92.6% specificity on validation but 43.2% on test, because its validation images were part of its training data.

---

## Explainability

High recall does not show *why* the model predicts pneumonia. Chest X-ray models are known to pick up shortcuts such as text markers, hospital labels or image borders instead of lung tissue, so the models are checked with two methods:

- **Grad-CAM** on the last convolutional block (`layer4`) highlights the regions that drove each prediction.
- **Occlusion sensitivity** hides one 15×15 patch at a time and measures the drop in pneumonia probability. It does not use gradients, so it cross-checks Grad-CAM.

Run from the repo root once the dataset is in `data/interim/chest_xray`:

```bash
python explain.py --image-size 150                         # ResNet-18 weights in this repo, whole test split
python explain.py --image-size 150 --image path/to/xray.jpeg   # the same model, a single X-ray
python explain.py --arch resnet50                          # ResNet-50 from models/baseline_resnet50.pth
```

Outputs:

| File | Description |
|--------|-------------|
| `reports/figures/explainability/<run>/<outcome>.png` | Grad-CAM examples for true/false positives and negatives |
| `reports/figures/explainability/<run>/occlusion_<outcome>.png` | Grad-CAM next to occlusion for the same images |
| `reports/explainability_summary_<run>.csv` | Per-image probability, prediction, outcome and `cam_border_fraction` |

`<run>` is the model name plus any `--image-size`, for example `resnet18_150`, so one model's report does not overwrite another's. The saved report for the weights in this repo is in `reports/explainability_summary.csv` and `reports/figures/explainability/`.

`cam_border_fraction` is the share of the heatmap in the outer 15% of the image. A flat heatmap scores about 0.51; values well above that point to border artefacts rather than the lungs. `explainability.ipynb` walks through the figures and records the findings.

Use `--image-size` when a model was trained at a different size from `src/dataset.py`. The ResNet-18 in this repo was trained at 150×150, and running it at 224×224 changes its results (183 false positives instead of 133).

### Explainability Findings (weights in this repo, test set, threshold 0.4)

| Outcome | Count | Mean border attention |
|---------|-------|-----------------------|
| True positive | 378 | 0.49 |
| True negative | 101 | 0.30 |
| False positive | 133 | 0.59 |
| False negative | 12 | 0.32 |

- Recall is 96.9%, but **specificity is only 43.2%**: 133 of 234 normal X-rays are flagged as pneumonia, most with high confidence.
- Grad-CAM heatmaps for pneumonia predictions are **diffuse** rather than focused on lung opacities.
- **False positives look outside the lungs** (image edges, background, below the diaphragm), with the highest border attention of any group.
- Normal predictions, including the 12 missed pneumonia cases, all focus on the upper-central chest and ignore the lower lung zones.
- `train_optimized.py` has since changed to address this. These findings still need to be checked on newly trained weights.

---

## Notes

- Datasets are **not pushed to GitHub**; only the folder structure is included.
- Run `setup.ipynb` to download the dataset, then the scripts above to train and evaluate. Trained weights for the newer ResNet-18 and the ResNet-50 are not in this repo yet.
- Users must download the dataset locally and place it in the appropriate `data/` directories.
- Generated reports can be found in the `reports/` folder.

