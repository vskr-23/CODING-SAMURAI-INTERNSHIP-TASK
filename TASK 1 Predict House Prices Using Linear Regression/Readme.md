# 🏡 Housing Price Prediction using Linear Regression

This project aims to predict house prices based on various features such as the number of bedrooms, bathrooms, square footage, and more. A Linear Regression model is built using Python and scikit-learn.

---

## 📁 Dataset Overview

The dataset contains **4,600** house listings with the following key features:

| Feature         | Description                        |
|----------------|------------------------------------|
| `price`         | Sale price of the house            |
| `bedrooms`      | Number of bedrooms                 |
| `bathrooms`     | Number of bathrooms                |
| `sqft_living`   | Interior living space in square feet |
| `sqft_lot`      | Total area of the property         |
| `floors`        | Number of floors                   |
| `waterfront`    | 1 if the house is on the waterfront, 0 otherwise |
| `view`          | Quality of the view                |
| `condition`     | Overall condition of the house     |

---

## 🔍 Data Analysis

- Checked for **missing values** — none found.
- Used `describe()` to understand distributions.
- Visualized feature correlations using a **heatmap**.
- Observed that `sqft_living` had the highest correlation with `price`.

---

## ⚙️ Model Building

- **Model Used:** `LinearRegression` from `sklearn.linear_model`
- **Features Selected:**
  ```python
  ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
   'waterfront', 'view', 'condition']

## 🧪 Sample Prediction

Predicted the price of a house with:

- 3 bedrooms  
- 2 bathrooms  
- 1500 sqft living  
- 4000 sqft lot  
- 1 floor  
- No waterfront  
- No view  
- Condition: 3

```python
new_data = [[3, 2, 1500, 4000, 1, 0, 0, 3]]
predicted_price = model.predict(new_data)
print("Predicted Price: $", round(predicted_price[0], 2))
```

## 📊 Evaluation Metrics

| Metric         | value                        |
|----------------|------------------------------------|
| `Mean Squarred Error`         | 986,869,414,954            |
| `R² Score`      |    `0.0323 (very low)`            |

Low R² suggests Linear Regression is not a good fit due to non-linearity or missing influential features.

## 📉 Visualizations
Actual vs Predicted Prices — Showed poor linear pattern.

Residual Plot — High and random residuals, further confirming poor fit.

## 🧪 Sample Prediction
Predicted the price of a house with:

3 bedrooms, 2 bathrooms, 1500 sqft living, 4000 sqft lot, 1 floor, no waterfront, no view, condition 3

```python
Predicted Price: $331,038.97
