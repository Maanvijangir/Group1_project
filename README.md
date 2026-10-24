# Computer Vision Project - 

## Overview
This project focuses on applying **Convolutional Neural Networks (CNNs)** for image-based disease detection.  
The workflow is designed to be reproducible, lightweight, and team-friendly.

## Dataset
- Source: Kaggle (Chest X-Ray dataset by Paul Mooney)
- Downloaded via **Kaggle API** in `setup.ipynb`
- zipped file in `data/raw/`
- UNzipped data in `data/interim`
- Preprocessing handled by `src/dataset.py` (transforms, normalization, dataloaders)

## Repository Structure
- src/ -> My python scripts(dataset.py for preprocessing and dataloaders)
- data/
   |- raw/ -> PLaceholder folder for Zipped file
   |- interim -> PLaceholder folder for Original dataset
   |- processed -> Placeholder folder for preocessed data
- setup.ipynb -> Notebook for Kaggle API integration, dataset unzip
- train.ipynb -> Notebook for training pipeline, calls get_loader function

## Notes 
- Datasets are **not pushed** to Github - only the folder structure is included.
- Code and notebooks are reproducible; datasets must be placed locally in 'data/raw'.
  
