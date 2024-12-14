# Area Sown Production Prediction Model

## Overview
This project involves building a machine learning model to predict crop yields based on area sown data. The model uses a Random Forest Regressor to predict the crop yield for the year 2019-20, leveraging historical area and production data.

## Objective
- Predict crop yield (target variable) based on area sown and other relevant features.
- Evaluate model performance using metrics like Mean Absolute Error (MAE) and R-squared (R²).

---

## Features

- **Data Preprocessing:**  
  - Cleaned and transformed the dataset, removing non-numeric values and handling missing data.
  - Encoded categorical features using Label Encoding.

- **Model Training and Evaluation:**  
  - Split data into training and testing sets.
  - Trained the Random Forest Regressor model on the training data and evaluated it on the test set.
  - Performance evaluated using R² score and Mean Absolute Error (MAE).

- **Model Accuracy:**  
  - **Training Set Accuracy:** 0.892
  - **Test Set Accuracy:** 0.956

---

## Setup

1. Clone or download the repository.
2. Install the required Python libraries:
   ```bash
   pip install pandas numpy scikit-learn matplotlib openpyxl
