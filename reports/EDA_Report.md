# Exploratory Data Analysis Report
## Chest X-ray Classification — Group 1

## 1. Introduction and Objectives

This report documents exploratory data analysis of chest X-ray
images belonging to the NORMAL and PNEUMONIA classes. The objective
is to examine dataset composition, image characteristics, and
data integrity before model development.

## 2. Dataset Source and Structure

Dataset: Chest X-Ray Images (Pneumonia)
Source: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

The dataset contains training, validation, and testing subsets.

## 3. Dataset Size and Class Distribution

The dataset contains 5,856 images:
- NORMAL: 1,583 (27.03%)
- PNEUMONIA: 4,273 (72.97%)

The class distribution is imbalanced and should be considered
when training and evaluating the classifier.

## 4. Dataset Split Analysis

- Training: 5,216 images
- Validation: 16 images
- Testing: 624 images

The validation set contains only eight images per class.
This is too small for a stable estimate of model performance.
The project team should review the validation strategy.

## 5. Sample Image Visualization

Representative images from both classes were inspected.
Visual examples illustrate the input data but do not establish
a medical diagnosis.

Figure: figures/sample_images.png

## 6. Image Dimensions and Color Modes

The mean image width is approximately 1,327.88 pixels, and
the mean height is approximately 970.69 pixels.

Color modes:
- Grayscale: 5,573 images
- RGB: 283 images

The variation in dimensions supports using a consistent
preprocessing pipeline. Resizing should be assessed carefully
because changing image proportions can distort anatomy.

Figure: figures/image_dimensions.png

## 7. Pixel Intensity and Aspect Ratio Analysis

The aspect-ratio results are:
- Mean: 1.443
- Median: 1.416
- Minimum: 0.835
- Maximum: 3.379

These results show variation in the original image proportions.
The preprocessing strategy should consider preserving the
aspect ratio where appropriate.

Overall pixel intensity analysis is descriptive and should not
be interpreted as a diagnostic test.

Figure: figures/pixel_intensity_distribution.png

## 8. Image Quality Analysis

The following class-wise results were obtained:

| Measurement | NORMAL mean | PNEUMONIA mean |
|---|---:|---:|
| Brightness | 122.65 | 122.84 |
| Contrast (standard deviation) | 60.77 | 54.99 |
| Sharpness (Laplacian variance) | 517.60 | 335.13 |

Average brightness is similar between the two classes in these
results. The average contrast and sharpness indicators differ.

These measurements are descriptive. Laplacian variance can be
affected by image content and noise and does not independently
establish medical image quality.

Figure: figures/image_quality_comparison.png

## 9. Duplicate Image Analysis

The exact-duplicate analysis identified 62 files involved in
duplicate groups. Zero exact duplicate groups were found to
span multiple dataset splits.

Therefore, exact duplicates exist within the dataset, but this
check did not identify exact file-content duplicates crossing
the train, validation, and test splits.

SHA-256 hashing does not detect every visually similar image,
different crop, or image belonging to the same patient.

## 10. Key Findings, Limitations, and Recommendations

- The dataset is class-imbalanced.
- The validation set is very small, containing only 16 images.
- Original image dimensions and color modes vary.
- Consistent image resizing, channel handling, and normalization
  are required.
- Class-wise precision, recall, F1-score, and a confusion matrix
  should be reported alongside accuracy.
- Review duplicate groups and investigate patient-level separation
  where patient identifiers are available.
- Review potentially unusual images rather than automatically
  deleting them based on arbitrary thresholds.
- Confirm that every referenced figure exists before submission.

## 11. Conclusion

The EDA establishes the dataset's composition and describes its
image dimensions, class distribution, image-quality indicators,
and duplicate-file findings. These results can guide preprocessing
and model evaluation.

The analysis is exploratory and is not a medical diagnostic assessment.
