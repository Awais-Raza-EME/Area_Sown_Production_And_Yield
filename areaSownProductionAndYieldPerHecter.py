import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

file_path = 'area-sown-production-and-yield.xlsx'
dataFile = pd.read_excel(file_path, header=0, index_col=0)  
dataFile.replace('**', pd.NA, inplace=True)
dataFile.dropna(inplace=True)  
dataFile.columns = dataFile.columns.str.strip()  # Remove leading/trailing whitespace
dataFile = dataFile.replace(',', '', regex=True)  # Remove commas from numeric columns
for col in dataFile.columns[1:]:  # Assuming the first column is categorical
    dataFile[col] = pd.to_numeric(dataFile[col], errors='coerce')
dataFile = dataFile.dropna()
# non_numeric_cols = dataFile.select_dtypes(exclude=['number']).columns
# if len(non_numeric_cols) > 0:
#     print("Non-numeric columns found:", non_numeric_cols.tolist())

# dataFile_encoded = pd.get_dummies(dataFile, columns=[dataFile.columns[0]], drop_first=True)
y = dataFile['2019-20']  # Your target variable
X = dataFile.drop(columns=['2019-20'])  # Drop target column to create features

label_encoder = LabelEncoder()
X[X.columns[0]] = label_encoder.fit_transform(X[X.columns[0]])

# # print(X.head())
# # print(y.head())
# # Ensure that all values in X are numeric
# # This will throw an error if there are still non-numeric values
# if not X.select_dtypes(include=['number']).shape[1] == X.shape[1]:
#     print("There are non-numeric values in the feature set.")
# else:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and fit the model
model = RandomForestRegressor(n_estimators=200, random_state=2)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
#mse = mean_squared_error(y_test, y_pred)
#rmse = mean_squared_error(y_test, y_pred, squared=False)  # RMSE
r2 = r2_score(y_test, y_pred)

# Print out the metrics
print("Mean Absolute Error (MAE):", mae)
#print("Mean Squared Error (MSE):", mse)
#print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R^2):", r2)

# print(f"Training set size: {X_train.shape[0]}")  # Should show the number of training samples
print(f"Test set size: {X_test.shape[0]}")  
print(X_test)
print(y_pred)
print("Accuracy on training set: {:.3f}".format(model.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(model.score(X_test, y_test)))



