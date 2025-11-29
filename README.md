# 🏡 House Price Prediction – Machine Learning Project

This project aims to build a **machine learning model** that predicts the price of a house based on multiple features such as area, number of bedrooms, bathrooms, furnishing status, location, etc.  
The model is trained using regression algorithms and helps understand how different factors influence house prices.

---

## 📂 Project Overview

In this project, we:

- Load and analyze the housing dataset  
- Clean and preprocess the data  
- Encode categorical variables  
- Split the dataset into training and testing sets  
- Train a regression model (Linear Regression)  
- Evaluate model performance  
- Predict house prices for new inputs  

---

## 📁 Dataset Information

The dataset contains the following sample features:

- **Area** – Size of the house (sq. ft)  
- **Bedrooms (BHK)**  
- **Bathrooms**  
- **Furnishing Status**  
- **Location**  
- **Parking**  
- **Age of Property**  
- **Price** (Target variable)

> Note: Features may differ slightly depending on the dataset.

---

## 🧠 Machine Learning Workflow

1. **Data Loading**  
2. **Data Cleaning**  
3. **Exploratory Data Analysis (EDA)**  
4. **Feature Encoding**  
5. **Model Training**  
6. **Evaluation (MAE, MSE, RMSE, R²)**  
7. **Predictions**

---

## 🛠 Technologies Used

- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  

---

## 🧪 Model Training Code

Below is the full code used in this project:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("House Price.csv.csv")

# Handle missing data
df.fillna(method='ffill', inplace=True)

# Encode categorical columns
categorical = df.select_dtypes(include='object').columns
le = LabelEncoder()
for col in categorical:
    df[col] = le.fit_transform(df[col])

# Split data
X = df.drop("Price", axis=1)
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R² Score:", r2)

# Plot Actual vs Predicted
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.show()
