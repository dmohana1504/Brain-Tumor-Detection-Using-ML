# Brain Tumor Detection Using a Convolutional Neural Network

## Introduction
This project demonstrates the power of Convolutional Neural Networks (CNNs) in medical imaging, particularly for detecting brain tumors from MRI images. Utilizing a Deep Learning algorithm, this application assigns importance to various aspects in the images, enabling it to differentiate between tumorous and non-tumorous brain MRIs.

## Dataset
The dataset comprises 253 Brain MRI Images divided into two categories: 155 tumorous (`yes`) and 98 non-tumorous (`no`). The dataset is available on Kaggle: [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection).

## Key Features

### Data Preprocessing
- Cropping technique to isolate brain region in images.
- Data augmentation to enhance model training.

### Model Architecture
- CNN model utilizing layers such as Conv2D, MaxPooling2D, Flatten, and Dense.
- Activation function: ReLU and Sigmoid for binary classification.

### Tools and Libraries
- TensorFlow and Keras for model building.
- OpenCV for image processing.
- Sklearn for data splitting and metrics calculation.

## Installation

### Prerequisites
- Python 3.11+
- TensorFlow 2.x
- OpenCV
- Sklearn

### Setup
Clone the repository and install the required packages:
```
git clone [repo-link]
cd [repo-name]
pip install -r requirements.txt
```

## Usage
1. Load the dataset.
2. Run the preprocessing script to crop and normalize the images.
3. Train the model using the provided script.
4. Evaluate the model's performance on the test dataset.

## Results
- Achieved an accuracy of 88.7% on the test set.
- F1 score of 0.88 on the test set.

## Future Work
- Implementing additional data augmentation techniques.
- Testing with larger datasets and different neural network architectures.
