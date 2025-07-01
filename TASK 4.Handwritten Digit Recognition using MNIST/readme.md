# MNIST Digit Classification: Binary Classification for Digit "2"

A machine learning project that performs binary classification on the famous MNIST handwritten digits dataset using Logistic Regression. The model specifically identifies whether a handwritten digit is a "2" or not (binary classification approach).

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Model Performance](#model-performance)
- [Visualizations](#visualizations)
- [Technical Implementation](#technical-implementation)
- [Results](#results)
- [Dependencies](#dependencies)

## 🔍 Overview

This project demonstrates binary classification using the MNIST dataset, focusing on distinguishing the digit "2" from all other digits (0,1,3,4,5,6,7,8,9). The approach showcases fundamental machine learning concepts including data preprocessing, model training, and cross-validation.

**Key Objectives:**
- Implement binary classification on MNIST dataset
- Achieve high accuracy in detecting digit "2"
- Demonstrate proper data handling and model evaluation techniques
- Visualize sample predictions and model performance

## ✨ Features

- **MNIST Dataset Integration**: Direct loading from sklearn's fetch_openml
- **Data Visualization**: Display handwritten digits with matplotlib
- **Binary Classification**: Convert multi-class problem to binary (digit 2 vs. others)
- **Logistic Regression**: Classical machine learning approach for binary classification
- **Cross-Validation**: Robust model evaluation using 3-fold cross-validation
- **Prediction Visualization**: Visual confirmation of model predictions
- **Data Shuffling**: Proper randomization for unbiased training

## 📊 Dataset

- **Source**: MNIST dataset via sklearn's fetch_openml('mnist_784')
- **Original Size**: 70,000 samples (60,000 training + 10,000 testing)
- **Working Subset**: 7,000 samples (6,000 training + 1,000 testing)
- **Features**: 784 pixel values (28×28 grayscale images)
- **Target Classes**: Originally 10 digits (0-9), converted to binary (2 vs. not-2)
- **Image Format**: 28×28 pixel grayscale images, flattened to 784-dimensional vectors

### Dataset Characteristics:
- **Pixel Values**: 0-255 (grayscale intensity)
- **Image Dimensions**: 28×28 pixels
- **Feature Vector Size**: 784 features per sample
- **Class Distribution**: Imbalanced (digit "2" is minority class)

## 🔬 Methodology

### 1. Data Loading and Exploration
```python
# Load MNIST dataset
mnist = fetch_openml('mnist_784')
x, y = mnist['data'], mnist['target']
```

### 2. Data Visualization
- Display sample handwritten digits
- Visualize digit "2" examples
- Show pixel intensity distributions

### 3. Data Preparation
- **Subset Selection**: Use 7,000 samples for computational efficiency
- **Train-Test Split**: 6,000 training, 1,000 testing samples
- **Data Shuffling**: Randomize training set for better learning
- **Target Transformation**: Convert multi-class to binary (digit 2 vs. others)

### 4. Binary Classification Setup
```python
# Convert to binary classification problem
y_train_2 = (y_train == 2)  # True for digit "2", False for others
y_test_2 = (y_test == 2)
```

### 5. Model Training
- **Algorithm**: Logistic Regression
- **Configuration**: max_iter=2000 for convergence
- **Training**: Fit model on binary-transformed labels

### 6. Model Evaluation
- **Cross-Validation**: 3-fold CV for robust evaluation
- **Metrics**: Accuracy score as primary metric
- **Prediction Testing**: Individual sample predictions

## 🤖 Technical Implementation

### Model Configuration
```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(max_iter=2000)
clf.fit(x_train, y_train_2)
```

### Key Parameters:
- **max_iter**: 2000 (increased for convergence on high-dimensional data)
- **solver**: Default 'lbfgs' (suitable for small datasets)
- **regularization**: Default L2 regularization

### Data Processing Pipeline:
1. **Raw Data**: 28×28 pixel images
2. **Flattening**: Convert to 784-dimensional vectors
3. **Normalization**: Pixel values already in 0-255 range
4. **Label Encoding**: Binary transformation (2 vs. not-2)
5. **Shuffling**: Random permutation of training data

## 📊 Model Performance

### Cross-Validation Results
- **Cross-Validation Folds**: 3-fold CV
- **Average Accuracy**: High accuracy expected (typically >95% for this task)
- **Evaluation Method**: `cross_val_score` with accuracy scoring

### Performance Characteristics:
- **Binary Classification**: Simplified from 10-class problem
- **Class Balance**: Imbalanced dataset (digit "2" is ~10% of data)
- **Feature Dimensionality**: 784 features per sample
- **Convergence**: Achieved within 2000 iterations

## 📊 Visualizations

The project generates several key visualizations:

1. **Sample Digit Display**: 28×28 pixel visualization of handwritten digits
2. **Prediction Examples**: Visual confirmation of model predictions
3. **Pixel Intensity Maps**: Grayscale representation of digit images
4. **Class Distribution**: Analysis of binary class balance


## 🔧 Model Architecture

### Logistic Regression Details:
- **Type**: Linear classifier for binary problems
- **Decision Boundary**: Linear separation in 784-dimensional space
- **Probability Output**: Sigmoid function for probability estimates
- **Training Algorithm**: Maximum likelihood estimation
- **Regularization**: L2 penalty (Ridge regression)

### Feature Space:
- **Input Dimensions**: 784 features (28×28 pixels)
- **Feature Range**: 0-255 (pixel intensities)
- **Feature Type**: Continuous numerical values
- **Preprocessing**: Minimal (data already well-structured)

## 📈 Results

### Key Findings:
- **Binary Classification Accuracy**: High performance on digit "2" detection
- **Model Convergence**: Successful training within iteration limit
- **Cross-Validation Stability**: Consistent performance across folds
- **Prediction Capability**: Accurate identification of individual samples

### Sample Prediction:
The model successfully predicts whether a given handwritten digit is a "2" or not, with visual confirmation available through matplotlib visualization.

## 🛠️ Dependencies

### Core Libraries
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computing and array operations
- `matplotlib`: Data visualization and plotting
- `scikit-learn`: Machine learning algorithms and utilities

## 💡 Learning Outcomes

### Machine Learning Concepts Demonstrated:
- **Binary Classification**: Reducing multi-class to binary problem
- **Logistic Regression**: Linear classification algorithm
- **Cross-Validation**: Robust model evaluation technique
- **Data Preprocessing**: Proper data handling and preparation
- **Visualization**: Effective data and result presentation

### Best Practices Implemented:
- **Data Shuffling**: Avoiding bias in training data
- **Train-Test Split**: Proper evaluation methodology
- **Parameter Tuning**: Adjusting max_iter for convergence
- **Code Organization**: Clean, readable implementation

## 🏆 Project Highlights

| Aspect | Details |
|--------|---------|
| **Dataset** | MNIST (70K samples, using 7K subset) |
| **Problem Type** | Binary Classification |
| **Algorithm** | Logistic Regression |
| **Evaluation** | 3-fold Cross Validation |
| **Features** | 784 (28×28 pixels) |
| **Target** | Digit "2" vs. Others |
