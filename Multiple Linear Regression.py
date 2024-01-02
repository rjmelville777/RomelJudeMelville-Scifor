#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Create a DataFrame with specific numbers
data = {
    'X1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'X2': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    'X3': [3, 6, 9, 12, 15, 18, 21, 24, 27, 30],
    'Y': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
}

df = pd.DataFrame(data)

# Add a column of ones for the intercept term
df['intercept'] = 1

# Define the feature columns (X) and the target column (Y)
X_columns = ['intercept', 'X1', 'X2', 'X3']
Y_column = 'Y'

# Convert DataFrame columns to numpy arrays
X = df[X_columns].values
Y = df[Y_column].values

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Function to perform multiple linear regression using gradient descent
def multiple_linear_regression(X, Y, learning_rate=0.01, epochs=1000):
    num_samples, num_features = X.shape
    weights = np.zeros(num_features)

    for epoch in range(epochs):
        predictions = np.dot(X, weights)
        errors = predictions - Y
        gradient = np.dot(errors, X) / num_samples
        weights -= learning_rate * gradient

        # Print the cost every 100 epochs
        if epoch % 100 == 0:
            cost = np.mean(errors**2)
            print(f'Epoch {epoch}, Cost: {cost}')

    return weights

# Train the model on the training set
learned_weights = multiple_linear_regression(X_train, Y_train)

# Test the model on the testing set
predictions_test = np.dot(X_test, learned_weights)

# Calculate metrics: MSE, RMSE, MAE, R-squared
mse = np.mean((predictions_test - Y_test)**2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(predictions_test - Y_test))
variance = np.var(Y_test, ddof=1)
r_squared = 1 - (mse / variance)

# Print the learned weights and metrics
print('Learned Weights:', learned_weights)
print('Mean Squared Error (MSE):', mse)
print('Root Mean Squared Error (RMSE):', rmse)
print('Mean Absolute Error (MAE):', mae)
print('R-squared:', r_squared)


# In[ ]:




