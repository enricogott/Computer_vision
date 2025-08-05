# Parking Slot Detector

This project focuses on detecting parking slots and analyzing their occupancy status using classical computer vision and machine learning techniques. Developed as part of the **Computer Vision course** at the **University of Padua**, it demonstrates two approaches for detecting and classifying the occupancy status of parking slots using the CNRPark dataset.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Methods](#methods)
  - [Parking Slot Detection](#parking-slot-detection)
  - [Parking Slot Occupancy Detection](#parking-slot-occupancy-detection)
    - [Approach 1: Color Histogram + ML](#approach-1-color-histogram--ml)
    - [Approach 2: SIFT + Bag of Words + ML](#approach-2-sift--bag-of-words--ml)
- [Results](#results)
- [Requirements](#requirements)
- [Usage](#usage)
- [Conclusion](#conclusion)

## Overview

The project is divided into two primary tasks:

1. **Parking Slot Detection**: Detecting parking slots in full-frame parking images using CSV metadata or edge detection.
2. **Occupancy Detection**: Determining whether a parking slot is busy or unoccupied using machine learning techniques.

## Dataset

The dataset used is from [CNRPark](http://cnrpark.it/) and includes:

- **CNR-EXT_FULL_IMAGE_1000x750**: Full images with parking lots.
- **CNRPark-Patches-150x150**: Cropped image patches of individual parking slots, labeled as `Busy` or `Unoccupied`.

## Project Structure

```
├── notebooks/
│   └── ParkingSlotDetector.ipynb      # Main Jupyter Notebook
├── data/
│   ├── CNR-EXT_FULL_IMAGE_1000x750/  # Full images + CSV coordinates
│   └── CNRPark-Patches-150x150/      # Labeled patches
├── results/
│   └── occupancy_predictions.csv     # Occupancy results per image
└── README.md
```

## Methods

### Parking Slot Detection

- **CSV-based Detection**: Used CSV files to draw rectangles around parking slots on full images.
- **Edge Detection (Optional)**: Experimented with edge detection and Hough transforms for unsupervised slot detection (not used in final version due to low accuracy).

### Parking Slot Occupancy Detection

Two approaches were implemented to classify the parking slot status:

#### Approach 1: Color Histogram + ML

- Calculated RGB histograms for each patch and used mean values as features.
- Machine learning models tested:
  - Logistic Regression
  - Linear Discriminant Analysis (LDA)
  - K-Nearest Neighbors (KNN)
  - Decision Tree (CART)
  - Gaussian Naive Bayes (NB)
  - Support Vector Machine (SVM)
- **Best Accuracy**: ~73% (KNN)

#### Approach 2: SIFT + Bag of Words + ML

- Applied SIFT feature extraction on each patch.
- Built Bag-of-Words (BoW) representation using KMeans clustering (k=20).
- Trained models on BoW features.
- **Best Accuracy**: ~76% (LDA)

## Results

| Approach       | Best Model | Accuracy |
|----------------|------------|----------|
| Histogram + ML| KNN        | ~73%     |
| SIFT + BoW + ML| LDA        | ~76%     |

> Note: SIFT-based method offered better accuracy but was computationally slower.

## Requirements

- Python 3.x
- Jupyter Notebook
- OpenCV
- NumPy
- Pandas
- scikit-learn
- Matplotlib

Install dependencies using:

```bash
pip install -r requirements.txt
```

## Usage

1. Clone the repository.
2. Download and extract the dataset from [http://cnrpark.it/](http://cnrpark.it/).
3. Run the notebook:

```bash
jupyter notebook notebooks/ParkingSlotDetector.ipynb
```

4. Modify paths in the notebook to point to your dataset location.

## Conclusion

- SIFT + BoW approach outperforms histogram-based approach in accuracy.
- CSV-based parking slot detection was sufficient, but slot coordinate inaccuracies in the dataset affected final accuracy.
- Further improvements can be made by using deep learning models for both detection and classification.
