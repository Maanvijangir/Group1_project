# Computer Vision Project - My work

## Overview
This project focuses on applying **Convolutional Neural Networks (CNNs)** for image-based disease detection.  
The workflow is designed to be reproducible, lightweight, and team-friendly.

## Dataset
- Source: Kaggle (Chest X-Ray dataset for pneumonia detection)
- Downloaded via **Kaggle API** in `setup.ipynb`
- Unzipped and stored in `data/raw/`
- Preprocessing handled by `src/dataset.py` (transforms, normalization, dataloaders)

## Repository Structure
   src/ -> My python scripts(dataset.py for preprocessing and dataloaders)
   data/
   |- raw/ -> PLaceholder folder for original dataset
   |- interim -> PLaceholder folder for intermediate dataset
   setup.ipynb -> Notebook for Kaggle API integration, dataset unzip
   train.ipynb -> Notebook for training pipeline, calls get_loader function

## Notes 
- Datasets are **not pushed** to Github - only the folder structure is included.
- Code and notebooks are reproducible; datasets must be placed locally in 'data/raw'.
  
