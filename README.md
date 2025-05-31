# Medical Image Classification using the Sperm Morphology Image Data Set (SMIDS)!

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A deep learning project for classifying sperm morphology images using Xception architecture with CBAM attention mechanism.

## Table of Contents
- Download :- [DataSet(SMIDS)](https://drive.google.com/drive/folders/1AKnwOSj1IXi8kg0oxDa7HOvNjKYi83Ue?usp=sharing)
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Technical Implementation](#technical-implementation)
  - [- Key Features](#key-features)
  - [- Performance Metrics](#performance-metrics)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Future Work](#future-work)

## Project Overview
This project implements a deep learning model to classify sperm morphology images into three categories:
- Normal_Sperm
- Abnormal_Sperm 
- Non-Sperm

The model uses a customized Xception architecture with CBAM (Convolutional Block Attention Module) attention mechanism to improve classification performance.

## Dataset
The Sperm Morphology Image Data Set (SMIDS) contains 3000 images distributed as:
- Normal_Sperm: 1021 images
- Abnormal_Sperm: 1005 images 
- Non-Sperm: 974 images
  
![Distribution of Tumor Types](https://github.com/Harsha-096/Medical-Image-Classification-using-the-Sperm-Morphology-Image-Dataset/blob/441a846a8c5e06a388896f04d85456526ba7855d/Reports/Distribution%20of%20Tumor%20Types.png)

## Technical Implementation

### - Key Features :
- Data preprocessing and class balancing using upsampling
- Transfer learning with ImageNet weights
- Detailed model evaluation metrics

### - Performance Metrics :
Test Accuracy: 90.55%
Detailed Classification Report:
```text
              precision    recall  f1-score   support

           0       0.76      0.82      0.79       102
           1       0.98      0.89      0.93       102
           2       0.83      0.83      0.83       103
    accuracy                           0.85       307
   macro avg       0.85      0.85      0.85       307
weighted avg       0.85      0.85      0.85       307
```
## Repository Structure
```text
/project-root
│── /notebooks
│   └── sperm_classification.ipynb  # Main implementation notebook
│── /reports
│   ├── Category Images.png        # Training curves
│   ├── Model Accuracy.png        # Training curves
│   ├── Model Loss.png        # Confusion matrix
│   └── Confusion Matrix.png   # Detailed metrics
│── README.md        
```
## Installation
1. Clone the repository:

git clone https://github.com/Harsha-096/Medical-Image-Classification-using-the-Sperm-Morphology-Image-Dataset.git

2. Install dependencies:

pip install -r requirements.txt
## Usage
1. Run the Jupyter notebook:
jupyter notebook notebooks/sperm_classification.ipynb

3. For training the model:
model.fit(train_gen_new, epochs=5, validation_data=valid_gen_new)

## Results
The model achieves:

Training accuracy: 93.61%

Validation accuracy: 89.87%

Test accuracy: 90.55%

Training curves and confusion matrix are available in the /reports directory.

## Future Work
Experiment with different attention mechanisms

Try other base architectures

Implement Grad-CAM for model interpretability

Deploy as a web application for clinical use
