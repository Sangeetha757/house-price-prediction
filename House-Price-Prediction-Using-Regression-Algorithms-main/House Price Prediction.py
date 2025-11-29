#!/usr/bin/env python
# coding: utf-8

# ## House Price preduction

# ## Aim
# The aim of this project is to develop a predictive model that accurately estimates residential property prices based on various features. The goal is to leverage machine learning algorithms and regression techniques to create a model capable of providing reliable predictions, enabling stakeholders in the real estate market to make informed decisions. The project seeks to enhance understanding of the key drivers behind property values and contribute to improved decision-making processes in real estate transactions.

# ## Overveiw
# In the context of the current real estate market, accurate price predictions are crucial for buyers, sellers, and real estate professionals to make informed decisions. The model will leverage machine learning algorithms and regression techniques, considering various property features such as bedrooms, bathrooms, square footage, and location. The project will involve data exploration, preprocessing, model development, and evaluation, with the ultimate goal of contributing valuable insights to the real estate domain.

# ## Project Scope

# Dataset Inclusions:
# - The project will utilize the provided dataset containing information on residential properties, including features such as bedrooms, bathrooms, square footage, and location.
# 
# Model Development:
# - The focus of the project is on developing machine learning models for predicting residential property prices.
# - Models will be selected from a range of regression techniques, including but not limited to linear regression, decision trees, random forests, and gradient boosting.
# 
# Feature Selection:
# - Feature selection will be performed to enhance the predictive power of the models.
# - Key features such as bedrooms, bathrooms, square footage, and location will be considered in the modeling process.

# ## Data Collection
# Source of Data:
# 
# - The dataset for this house price prediction project is sourced from the Kaggle platform. It is available in the Kaggle dataset repository under the title [housr price preduction].
# 1. **date:** Represents the recording date or transaction date of the property data.
# 
# 2. **price:** Indicates the property price in a specific currency (e.g., dollars).
# 
# 3. **bedrooms:** Shows the number of bedrooms in the property.
# 
# 4. **bathrooms:** Indicates the number of bathrooms in the property.
# 
# 5. **sqft_living:** Represents the total living space square footage.
# 
# 6. **sqft_lot:** Indicates the total lot or land square footage.
# 
# 7. **floors:** Indicates the number of floors in the property.
# 
# 8. **waterfront:** Binary indicator (0 or 1) for waterfront view presence.
# 
# 9. **view:** Represents a rating or indicator of the property's view quality.
# 
# 10. **condition:** Indicates the overall property condition, possibly on a numerical scale.
# 
# 11. **sqft_above:** Represents the square footage of the above-ground living space.
# 
# 12. **sqft_basement:** Represents the square footage of the basement, if applicable.
# 
# 13. **yr_built:** Indicates the year of the property's original construction.
# 
# 14. **yr_renovated:** Represents the year of the last renovation; 0 if never renovated.
# 
# 15. **street:** Contains the street address of the property.
# 
# 16. **city:** Represents the city where the property is located.
# 
# 17. **statezip:** Combines state and ZIP code information for the property.
# 
# 18. **country:** Indicates the country where the property is located.

# In[1]:


import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from math import radians, sin, cos, sqrt, atan2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import time


# In[2]:


file_path = 'data.csv'
df = pd.read_csv(file_path)
print(df.head())


# In[3]:


df.info()


# In[4]:


df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df = df.drop('date', axis=1)
df.to_csv('House Price Preduction.csv', index=False)


# In[5]:


print(df['year'])


# In[6]:


df.tail()


# In[7]:


df.describe()


# In[8]:


categorical_columns = df.select_dtypes(include=['object']).columns
for column in categorical_columns:
    unique_values = df[column].unique()
    print(f"Unique values in {column}: {unique_values}")


# In[9]:


column_of_interest = 'bedrooms'
value_counts = df[column_of_interest].value_counts()
print(f"Value counts for {column_of_interest}:\n{value_counts}")


# ## Data Cleaning, Analysis and Visualization

# In[10]:


df.isna().sum()


# In[11]:


df.dropna(inplace=True)


# In[12]:


df.shape


# In[13]:


# Remove outliers 
df['price_zscore'] = zscore(df['price'])
df['sqft_living_zscre'] = zscore(df['sqft_living'])
df['sqft_lot_zscre'] = zscore(df['sqft_lot'])
df['sqft_above_zscre'] = zscore(df['sqft_above'])
df['sqft_basement_zscre'] = zscore(df['sqft_basement'])

df = df[(df.price_zscore < 3) & (df.price_zscore > -3)]
df = df[(df.sqft_living_zscre < 3) & (df.sqft_living_zscre > -3)]
df = df[(df.sqft_lot_zscre < 3) & (df.sqft_lot_zscre > -3)]
df = df[(df.sqft_above_zscre < 3) & (df.sqft_above_zscre > -3)]
df = df[(df.sqft_basement_zscre < 3) & (df.sqft_basement_zscre > -3)]


# In[14]:


df.drop(columns=['price_zscore','sqft_living_zscre','sqft_lot_zscre','sqft_above_zscre','sqft_basement_zscre'], inplace=True)


# In[15]:


df.shape


# In[16]:


df.describe()


# ## Remove unused features

# In[17]:


df.country.nunique()


# In[18]:


df.city.nunique()


# In[19]:


sns.set(style="whitegrid")

plt.figure(figsize=(10, 6))
sns.histplot(df['sqft_living'], bins=30, kde=True, color='green')
plt.title('Distribution of Square Footage of Living Area')
plt.xlabel('Square Footage of Living Area')


plt.ylabel('Frequency')
plt.show()


# In[20]:


plt.figure(figsize=(10, 6))
sns.countplot(x='bedrooms', data=df)
plt.title('Number of Bedrooms with Number of Houses')
plt.xlabel('Number of Bedrooms')
plt.ylabel('Number of Houses')
plt.show()


# In[21]:


plt.figure(figsize=(10, 6))
sns.countplot(x='bathrooms', data=df)
plt.title('Number of Bathrooms with Number of Houses')
plt.xlabel('Number of Bathrooms')
plt.ylabel('Number of Houses')
plt.show()


# In[22]:


plt.figure(figsize=(12, 8))
sns.scatterplot(x='sqft_living', y='price', data=df)
plt.title('Living Square Footage vs Price')
plt.xlabel('Living Square Footage')
plt.ylabel('Price')
plt.show()


# In[23]:


plt.figure(figsize=(12, 8))
sns.scatterplot(x='sqft_lot', y='price', data=df)
plt.title('Lot Square Footage vs Price')
plt.xlabel('Lot Square Footage')
plt.ylabel('Price')
plt.show()


# In[24]:


plt.figure(figsize=(10, 6))
sns.countplot(x='floors', data=df)
plt.title('Number of Floors with Number of Houses')
plt.xlabel('Number of Floors')
plt.ylabel('Number of Houses')
plt.show()


# In[25]:


waterfront_counts = df['waterfront'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(waterfront_counts, labels=waterfront_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Houses with Waterfront')
plt.show()


# In[26]:


plt.figure(figsize=(10, 6))
sns.countplot(x='view', data=df)
plt.title('View with Number of Houses')
plt.xlabel('View')
plt.ylabel('Number of Houses')
plt.show()


# In[27]:


plt.figure(figsize=(10, 6))
sns.countplot(x='condition', data=df)
plt.title('Condition with Number of Houses')
plt.xlabel('Condition')
plt.ylabel('Number of Houses')
plt.show()


# In[28]:


plt.figure(figsize=(12, 8))
sns.scatterplot(x='sqft_above', y='price', data=df)
plt.title('Above Square Footage vs Price')
plt.xlabel('Above Square Footage')
plt.ylabel('Price')
plt.show()


# In[29]:


plt.figure(figsize=(12, 8))
sns.scatterplot(x='sqft_basement', y='price', data=df)
plt.title('Basement Square Footage vs Price')
plt.xlabel('Basement Square Footage')
plt.ylabel('Price')
plt.show()


# In[30]:


plt.figure(figsize=(15, 6))
sns.countplot(x='yr_built', data=df)
plt.title('Year Built with Number of Houses')
plt.xlabel('Year Built')
plt.ylabel('Number of Houses')
plt.xticks(rotation=90)
plt.show()


# In[31]:


plt.figure(figsize=(15, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Heat Map for the Entire Dataset')
plt.show()


# ## Data Preprocessing

# In[32]:


df.isna().sum()


# In[33]:


df.head()


# In[34]:


pd.set_option('display.max_columns', None)
df.head()


# In[35]:


# Check and encode categorical variables using one-hot encoding
categorical_columns = df.select_dtypes(include=['object']).columns
df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)


# One-Hot Encoding
# - One-hot encoding is a technique used in machine learning to represent categorical variables as binary vectors. Categorical variables are those that can take on a limited, fixed number of values, often representing different classes or categories. One-hot encoding is particularly useful when working with algorithms that require numerical input, as it transforms categorical data into a format that is compatible with these models.
# 
# How One-Hot Encoding Works:
# - In one-hot encoding, each unique category in a categorical variable is represented as a binary (0 or 1) vector. For each observation in the dataset, only one element in the vector is set to 1, corresponding to the category of that observation. All other elements are set to 0. This creates a binary matrix where each column represents a unique category.

# In[36]:


numeric_columns = df_encoded.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()
df_scaled = df_encoded.copy()
df_scaled[numeric_columns] = scaler.fit_transform(df_encoded[numeric_columns])


# In[37]:


df['total_sqft'] = df['sqft_living'] + df['sqft_above'] + df['sqft_basement']
print(df['total_sqft'])


# ## Price Preduction models

# ## Linear Regression

# In[68]:


df_numeric = df.drop(['statezip', 'street', 'city', 'country'], axis=1)

# Select features for training
selected_features = ['sqft_living', 'sqft_lot', 'floors', 'bedrooms', 'bathrooms']

# Split the data into features (X) and target variable (y)
X = df_numeric[selected_features]
y = df_numeric['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)

# Feature Scaling using Min-Max scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Get user input for features and scale them
living_area = float(input("Enter the living area: "))
lot_area = float(input("Enter the lot area: "))
num_floors = float(input("Enter the number of floors: "))
num_bedrooms = int(input("Enter the number of bedrooms: "))
num_bathrooms = float(input("Enter the number of bathrooms: "))

# Create a DataFrame with user input
user_input_df = pd.DataFrame({
    'sqft_living': [living_area],
    'sqft_lot': [lot_area],
    'floors': [num_floors],
    'bedrooms': [num_bedrooms],
    'bathrooms': [num_bathrooms]
})

# Ensure feature order matches the order during training
user_input_df = user_input_df[selected_features]

# Scale user input
user_input_scaled = scaler.transform(user_input_df)

# Define threshold for binary classification
threshold = 500000  # Adjust threshold as needed

# Training the linear regression model with scaled features
linear_reg_scaled = LinearRegression()
start_time = time.time()
linear_reg_scaled.fit(X_train_scaled, y_train)
end_time = time.time()

# Testing the linear regression model
start_test_time = time.time()
linear_reg_preds_scaled = linear_reg_scaled.predict(X_test_scaled)
end_test_time = time.time()

# Calculate percentages
total_execution_time = end_test_time - start_time
training_percentage = ((end_time - start_time) / total_execution_time) * 100
testing_percentage = ((end_test_time - start_test_time) / total_execution_time) * 100

# Predict the price for user input with scaled features
predicted_price_scaled = linear_reg_scaled.predict(user_input_scaled)

# Print the predicted price
print(f"Predicted Price: ${predicted_price_scaled[0]:,.2f}")

# Check the actual price for the user input in the test set
actual_price = df.loc[
    (df['sqft_living'] == living_area) &
    (df['sqft_lot'] == lot_area) &
    (df['floors'] == num_floors) &
    (df['bedrooms'] == num_bedrooms) &
    (df['bathrooms'] == num_bathrooms),
    'price'
].values

# Print the actual price
if len(actual_price) > 0:
    print(f"Actual Price: ${actual_price[0]:,.2f}")
else:
    print("No matching data found in the test set.")

# Calculate accuracy for linear regression model
accuracy = linear_reg_scaled.score(X_test_scaled, y_test)
print(f"Linear Regression with Scaling - Accuracy: {accuracy * 100:.2f}")

# Calculate confusion matrix for linear regression model
conf_matrix = confusion_matrix((y_test > threshold).astype(int), (linear_reg_preds_scaled > threshold).astype(int))

# Print confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

# Convert predictions to binary labels (Above/Below a threshold)
binary_preds = (linear_reg_preds_scaled > threshold).astype(int)
binary_y_test = (y_test > threshold).astype(int)

# Calculate F1 score for linear regression model
f1 = f1_score(binary_y_test, binary_preds)
print(f"Linear Regression with Scaling - F1 Score: {f1:.2f}")

# Calculate precision and recall for linear regression model
precision_linear_reg = precision_score(binary_y_test, binary_preds)
recall_linear_reg = recall_score(binary_y_test, binary_preds)

print(f"Linear Regression with Scaling - Precision: {precision_linear_reg:.2f}")
print(f"Linear Regression with Scaling - Recall: {recall_linear_reg:.2f}")

# Print execution times
# Print percentages
print(f"Training Percentage: {training_percentage:.2f}%")
print(f"Testing Percentage: {testing_percentage:.2f}%")
print(f"Total Execution Time: {total_execution_time:.4f} seconds")


# ## RandomForest

# In[52]:


df_numeric = df.drop(['statezip', 'street', 'city', 'country'], axis=1)

# Select features for training
selected_features = ['sqft_living', 'sqft_lot', 'floors', 'bedrooms', 'bathrooms']

# Split the data into features (X) and target variable (y)
X = df_numeric[selected_features]
y = df_numeric['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)

# Feature Scaling using Min-Max scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Get user input for features and scale them
living_area = float(input("Enter the living area: "))
lot_area = float(input("Enter the lot area: "))
num_floors = float(input("Enter the number of floors: "))
num_bedrooms = int(input("Enter the number of bedrooms: "))
num_bathrooms = float(input("Enter the number of bathrooms: "))

# Create a DataFrame with user input
user_input_df = pd.DataFrame({
    'sqft_living': [living_area],
    'sqft_lot': [lot_area],
    'floors': [num_floors],
    'bedrooms': [num_bedrooms],
    'bathrooms': [num_bathrooms]
})

# Ensure feature order matches the order during training
user_input_df = user_input_df[selected_features]

# Scale user input
user_input_scaled = scaler.transform(user_input_df)

# Training the Random Forest model with scaled features
random_forest_scaled = RandomForestRegressor()
start_time = time.time()
random_forest_scaled.fit(X_train_scaled, y_train)
end_time = time.time()

# Testing the Random Forest model
start_test_time = time.time()
random_forest_preds_scaled = random_forest_scaled.predict(X_test_scaled)
end_test_time = time.time()

# Calculate percentages
total_execution_time = end_test_time - start_time
training_percentage = ((end_time - start_time) / total_execution_time) * 100
testing_percentage = ((end_test_time - start_test_time) / total_execution_time) * 100

# Predict the price for user input with scaled features
predicted_price_rf_scaled = random_forest_scaled.predict(user_input_scaled)

# Print the predicted price
print(f"Predicted Price: ${predicted_price_rf_scaled[0]:,.2f}")

# Check the actual price for the user input in the test set
actual_price = df.loc[
    (df['sqft_living'] == living_area) &
    (df['sqft_lot'] == lot_area) &
    (df['floors'] == num_floors) &
    (df['bedrooms'] == num_bedrooms) &
    (df['bathrooms'] == num_bathrooms),
    'price'
].values

# Print the actual price
if len(actual_price) > 0:
    print(f"Actual Price: ${actual_price[0]:,.2f}")
else:
    print("No matching data found in the test set.")

# Calculate accuracy
accuracy = random_forest_scaled.score(X_test_scaled, y_test)
print(f"Random Forest with Scaling - Accuracy: {accuracy * 100:.2f}")

# Calculate confusion matrix
threshold = 500000  # Adjust threshold as needed
conf_matrix = confusion_matrix((y_test > threshold).astype(int), (random_forest_preds_scaled > threshold).astype(int))

# Print confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

# Convert predictions to binary labels (Above/Below a threshold)
binary_preds = (random_forest_preds_scaled > threshold).astype(int)
binary_y_test = (y_test > threshold).astype(int)

# Calculate F1 score
f1 = f1_score(binary_y_test, binary_preds)
print(f"Random Forest with Scaling - F1 Score: {f1:.2f}")

# Calculate recall
recall_rf_scaled = recall_score(binary_y_test, binary_preds)
print(f"Random Forest with Scaling - Recall: {recall_rf_scaled:.2f}")

# Print execution times
print(f"Training Percentage: {training_percentage:.2f}%")
print(f"Testing Percentage: {testing_percentage:.2f}%")
print(f"Total Execution Time: {total_execution_time:.4f} seconds")


# ## Gradient Boosting

# In[53]:


df_numeric = df.drop(['statezip', 'street', 'city', 'country'], axis=1)

# Select features for training
selected_features = ['sqft_living', 'sqft_lot', 'floors', 'bedrooms', 'bathrooms']

# Split the data into features (X) and target variable (y)
X = df_numeric[selected_features]
y = df_numeric['price']

# Split the data into training and testing sets wi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)

# Feature Scaling using Min-Max scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Get user input for features and scale them
living_area = float(input("Enter the living area: "))
lot_area = float(input("Enter the lot area: "))
num_floors = float(input("Enter the number of floors: "))
num_bedrooms = int(input("Enter the number of bedrooms: "))
num_bathrooms = float(input("Enter the number of bathrooms: "))

# Create a DataFrame with user input
user_input_df = pd.DataFrame({
    'sqft_living': [living_area],
    'sqft_lot': [lot_area],
    'floors': [num_floors],
    'bedrooms': [num_bedrooms],
    'bathrooms': [num_bathrooms]
})

# Ensure feature order matches the order during training
user_input_df = user_input_df[selected_features]

# Scale user input
user_input_scaled = scaler.transform(user_input_df)

# Training the Gradient Boosting Regressor model with scaled features
gradient_boosting = GradientBoostingRegressor()
start_time = time.time()
gradient_boosting.fit(X_train_scaled, y_train)
end_time = time.time()

# Testing the Gradient Boosting Regressor model
start_test_time = time.time()
gradient_boosting_preds = gradient_boosting.predict(X_test_scaled)
end_test_time = time.time()

# Calculate percentages
total_execution_time = end_test_time - start_time
training_percentage = ((end_time - start_time) / total_execution_time) * 100
testing_percentage = ((end_test_time - start_test_time) / total_execution_time) * 100

# Predict the price for user input with scaled features
predicted_price_gb_scaled = gradient_boosting.predict(user_input_scaled)

# Print the predicted price
print(f"Predicted Price: ${predicted_price_gb_scaled[0]:,.2f}")

# Check the actual price for the user input in the test set
actual_price = df.loc[
    (df['sqft_living'] == living_area) &
    (df['sqft_lot'] == lot_area) &
    (df['floors'] == num_floors) &
    (df['bedrooms'] == num_bedrooms) &
    (df['bathrooms'] == num_bathrooms),
    'price'
].values

# Print the actual price
if len(actual_price) > 0:
    print(f"Actual Price: ${actual_price[0]:,.2f}")
else:
    print("No matching data found in the test set.")

# Calculate accuracy
accuracy = gradient_boosting.score(X_test_scaled, y_test)
print(f"Gradient Boosting with Scaling - Accuracy: {accuracy * 100:.2f}")

# Calculate confusion matrix
conf_matrix = confusion_matrix((y_test > threshold).astype(int), (gradient_boosting_preds > threshold).astype(int))

# Print confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

# Calculate Precision, Recall, and F1 score
precision_gb_scaled = precision_score((y_test > threshold).astype(int), (gradient_boosting_preds > threshold).astype(int))
recall_gb_scaled = recall_score((y_test > threshold).astype(int), (gradient_boosting_preds > threshold).astype(int))
f1_gb_scaled = f1_score((y_test > threshold).astype(int), (gradient_boosting_preds > threshold).astype(int))

print(f"Gradient Boosting with Scaling - Precision: {precision_gb_scaled:.2f}")
print(f"Gradient Boosting with Scaling - Recall: {recall_gb_scaled:.2f}")
print(f"Gradient Boosting with Scaling - F1 Score: {f1_gb_scaled:.2f}")

# Print execution times
print(f"Training Percentage: {training_percentage:.2f}%")
print(f"Testing Percentage: {testing_percentage:.2f}%")
print(f"Total Execution Time: {total_execution_time:.4f} seconds")


# ## SVM

# In[62]:


df_numeric = df.drop(['statezip', 'street', 'city', 'country'], axis=1)

# Select features for training
selected_features = ['sqft_living', 'sqft_lot', 'floors', 'bedrooms', 'bathrooms']

# Split the data into features (X) and target variable (y)
X = df_numeric[selected_features]
y = df_numeric['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling using Min-Max scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Get user input for features and scale them
living_area = float(input("Enter the living area: "))
lot_area = float(input("Enter the lot area: "))
num_floors = float(input("Enter the number of floors: "))
num_bedrooms = int(input("Enter the number of bedrooms: "))
num_bathrooms = float(input("Enter the number of bathrooms: "))

# Create a DataFrame with user input
user_input_df = pd.DataFrame({
    'sqft_living': [living_area],
    'sqft_lot': [lot_area],
    'floors': [num_floors],
    'bedrooms': [num_bedrooms],
    'bathrooms': [num_bathrooms]
})

# Ensure feature order matches the order during training
user_input_df = user_input_df[selected_features]

# Scale user input
user_input_scaled = scaler.transform(user_input_df)

# Training the Support Vector Regressor (SVR) model
svm_regressor = SVR(kernel='linear')
start_time = time.time()
svm_regressor.fit(X_train_scaled, y_train)
end_time = time.time()

# Testing the SVR model
start_test_time = time.time()
svm_preds = svm_regressor.predict(X_test_scaled)
end_test_time = time.time()

# Calculate percentages
total_execution_time = end_test_time - start_time
training_percentage = ((end_time - start_time) / total_execution_time) * 100
testing_percentage = ((end_test_time - start_test_time) / total_execution_time) * 100

# Predict the price for user input with scaled features
predicted_price_svm_scaled = svm_regressor.predict(user_input_scaled)

# Print the predicted price
print(f"Predicted Price: ${predicted_price_svm_scaled[0]:,.2f}")

# Check the actual price for the user input in the test set
actual_price = df.loc[
    (df['sqft_living'] == living_area) &
    (df['sqft_lot'] == lot_area) &
    (df['floors'] == num_floors) &
    (df['bedrooms'] == num_bedrooms) &
    (df['bathrooms'] == num_bathrooms),
    'price'
].values

# Print the actual price
if len(actual_price) > 0:
    print(f"Actual Price: ${actual_price[0]:,.2f}")
else:
    print("No matching data found in the test set.")

# Calculate accuracy
accuracy = svm_regressor.score(X_test_scaled, y_test)
print(f"SVM Regression with Scaling - Accuracy: {accuracy * 100:.2f}%")

# Calculate confusion matrix
threshold = 500000  # Adjust threshold as needed
binary_preds = (svm_preds > threshold).astype(int)
binary_y_test = (y_test > threshold).astype(int)
conf_matrix = confusion_matrix(binary_y_test, binary_preds)

# Print confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

# Calculate recall, precision, and F1 score
recall = recall_score(binary_y_test, binary_preds)
precision = precision_score(binary_y_test, binary_preds)
f1 = f1_score(binary_y_test, binary_preds)

print(f"SVM Regression with Scaling - Recall: {recall:.2f}")
print(f"SVM Regression with Scaling - Precision: {precision:.2f}")
print(f"SVM Regression with Scaling - F1 Score: {f1:.2f}")

# Print execution times
print(f"Training Percentage: {training_percentage:.2f}%")
print(f"Testing Percentage: {testing_percentage:.2f}%")
print(f"Total Execution Time: {total_execution_time:.4f} seconds")


# ## About Outputs

# Linear Regression:
# - The linear regression model predicted a house price of $493,239.97$ for a property with specific features, while the actual price was $342,000.00. $The model's performance metrics indicate moderate accuracy. The Mean Squared Error (MSE) of $50.93 $billion and Mean Absolute Error (MAE) of $161,289.58 $reveal the average squared and absolute differences, respectively, between predicted and actual prices. The R-squared (R²) value of 0.38 indicates that approximately 38% of the variability in house prices is explained by the model. The Explained Variance Score supports this, also at 0.38. The training and testing percentages suggest a relatively balanced model, trained on 74.63% of the data and tested on 25.37%.
# 
# Random Forest:
# - The random forest model predicted a house price of $357,083.50, $while the actual price was $342,000.00. $The MSE and MAE are $54.62 $billion and $161,423.73, $respectively, indicating a moderate deviation from actual prices. The R² of 0.33 suggests that 33% of the variability in house prices is explained by the model, with the Explained Variance Score supporting this at 0.33. The model is trained on 88.09% of the data and tested on 11.91%.
# 
# Gradient Boosting:
# - For the gradient boosting model, the predicted house price was $393,588.55, $compared to the actual price of $342,000.00. $The MSE and MAE are $53.95 $billion and $159,277.59, $respectively. The R² of 0.34 suggests that 34% of the variability in house prices is explained by the model, supported by an Explained Variance Score of 0.34. The model is trained on 92.19% of the data and tested on 7.81%.
# 
# SVM Regression:
# - The SVM regression model predicted a house price of $454,813.09, $while the actual price was $342,000.00. $However, the model's performance metrics are less favorable, with a high MSE of $78.10 $billion and MAE of $201,093.48. $The negative R² (-0.03) suggests a poor fit, worse than a model predicting the mean. The Explained Variance Score is 0.00, reinforcing the poor fit. The model is trained on 86.73% of the data and tested on 13.27%.
# 

# ## Conclusion

# In summary the main goal of this project was to use machine learning techniques to predict property prices. The process involved exploring and analyzing data, developing and evaluating models. Notable accomplishments included thorough data preprocessing creating features and applying regression techniques, like linear regression, decision trees, random forests and gradient boosting. By focusing on attributes such as bedrooms, bathrooms, square footage and location in feature selection the accuracy of predictions was significantly improved. The dataset used from Kaggle provided a foundation for analysis. This project not revealed insights into the factors influencing property prices. Also provided guidance for future considerations like optimizing the models expanding the set of features used real world implementation possibilities and continuous improvement. As we wrap up this phase of the project our developed models are now ready, for deployment to facilitate decision making in the ever changing world of real estate transactions.
# 

# In[ ]:




