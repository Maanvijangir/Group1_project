# Computer Vision Project - Pneumonia Detection Using CNN

## Overview

This project focuses on applying **Convolutional Neural Networks (CNNs)** for chest X-ray image classification and pneumonia detection. The workflow is designed to be reproducible, lightweight, and team-friendly, enabling easy dataset setup, preprocessing, training, and evaluation.

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
│   ├── eda_summary.csv
│   ├── image_aspect_ratio_summary.csv
│   ├── image_metadata.csv
│   └── ...
│
├── src/
│   └── dataset.py    # Preprocessing and dataloader implementation
│   └── train.py
|
├── EDA.ipynb         # Exploratory Data Analysis
├── setup.ipynb       # Kaggle API integration and dataset extraction
├── train.ipynb       # Training notebook
├── train_optimized.py # Optimized model training script
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
| `eda_summary.csv` | Summary statistics and EDA findings |
| `image_aspect_ratio_summary.csv` | Aspect ratio distribution analysis |
| `image_metadata.csv` | Extracted image metadata |
| Other report files | Additional dataset analysis outputs |

---

## Model Training

The optimized CNN model is trained using the pipeline implemented in:

```text
train_optimized.py
```

This script handles:

- Dataset loading
- Data preprocessing
- Model training
- Validation
- Performance evaluation

---

## Results

### Model Performance

| Phase | Metric | Score |
|---------|---------|---------|
| Training | Average Loss | 0.2346 |
| Validation | Recall | 100.00% |
| Testing | Recall | 96.92% |

### Key Findings

- Low training loss (**0.2346**) indicates successful model learning.
- Achieved **100% recall on the validation set**, demonstrating excellent pneumonia detection capability during validation.
- Achieved **96.92% recall on the test set**, showing strong generalization to unseen data.
- The model prioritizes high recall, which is critical in medical diagnosis tasks where missing pneumonia cases can have serious consequences.

---

## Notes

- Datasets are **not pushed to GitHub**; only the folder structure is included.
- The project is fully reproducible using the provided notebooks and scripts.
- Users must download the dataset locally and place it in the appropriate `data/` directories.
- Generated reports can be found in the `reports/` folder.

