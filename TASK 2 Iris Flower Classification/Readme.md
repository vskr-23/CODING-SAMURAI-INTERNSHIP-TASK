# Iris Flower Classification using K-Nearest Neighbors (KNN)

This project uses the famous Iris dataset to build a classification model that predicts the species of an Iris flower based on its sepal and petal measurements. The model is implemented using the K-Nearest Neighbors (KNN) algorithm from Scikit-Learn.

## 📊 Dataset

- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data)
- **Features:**
  - Sepal Length
  - Sepal Width
  - Petal Length
  - Petal Width
- **Target:** Iris species (`Iris-setosa`, `Iris-versicolor`, `Iris-virginica`)

## 📌 Libraries Used

- `pandas` – for data loading and manipulation
- `matplotlib` and `seaborn` – for visualization
- `scikit-learn` – for machine learning modeling and evaluation

## 🧪 Steps Performed

1. **Data Loading and Exploration:**
   - Loaded the Iris dataset using `pandas`.
   - Displayed basic statistics and visualized pair plots using `seaborn`.

2. **Preprocessing:**
   - Separated features (`X`) and target labels (`y`).
   - Split the data into training and testing sets (70/30 split).

3. **Model Training:**
   - Trained a `KNeighborsClassifier` with `n_neighbors=3`.

4. **Model Evaluation:**
   - Evaluated the classifier using `accuracy_score` and `classification_report`.
   - Achieved **100% accuracy** on the test set.

5. **Prediction:**
   - Predicted the species for a new Iris flower with:
     ```python
     sepal_length: 6.7
     sepal_width: 6.5
     petal_length: 3.4
     petal_width: 2.9
     ```
   - **Predicted class:** `Iris-virginica`

## ✅ Output Summary

- **Accuracy:** `1.0`
- **Precision/Recall/F1-score:** Perfect scores for all classes.
- **Model used:** `KNeighborsClassifier(n_neighbors=3)`

## 📁 How to Run

1. Clone the repository.
2. Make sure you have Python and required libraries installed:
   ```bash
   pip install pandas matplotlib seaborn scikit-learn

