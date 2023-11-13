# Brain Tumor Detection and Data Augmentation Project

## Overview

This repository contains two Jupyter notebooks detailing a project focused on detecting brain tumors using Convolutional Neural Networks (CNN) and the implementation of data augmentation techniques to enhance the model's performance.

### 1. Brain Tumor Detection Using a Convolutional Neural Network

The `Brain Tumor Detection.ipynb` notebook outlines the process of using a CNN to detect brain tumors from MRI images. The dataset consists of 253 MRI images, divided into two categories: 'yes' (155 images with tumors) and 'no' (98 images without tumors). The notebook includes steps like data preparation, preprocessing, model training, and evaluation.

Key Highlights:
- Cropping technique to isolate the brain from MRI images.
- Building and training a CNN model.
- Evaluation metrics including accuracy and F1 score on test data.

### 2. Data Augmentation

The `Data Augmentation.ipynb` notebook describes the use of data augmentation techniques to address the small size and imbalance of the dataset. This process generates additional synthetic data, helping to improve the robustness of the CNN model.

Key Highlights:
- Addressing data imbalance between 'yes' and 'no' classes.
- Generating new images to augment the dataset.
- Preparing the augmented data for CNN training.

## Dataset

The Brain MRI Images for Brain Tumor Detection dataset is used in both notebooks. It can be found at [Kaggle](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection).

## Requirements

- Python 3.x
- Libraries: [List all necessary libraries, e.g., TensorFlow, NumPy, OpenCV, etc.]

## Usage

To use these notebooks:
1. Clone the repository.
2. Install the required libraries.
3. Run the notebooks in a Jupyter environment.

## Conclusion

These notebooks demonstrate the power of CNNs in medical image analysis and the effectiveness of data augmentation in dealing with imbalanced and limited data.
