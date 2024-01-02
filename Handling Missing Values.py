#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

data = {'Name': ['Alice', 'Bob', 'Charlie', 'David', np.nan],
        'Age': [25, np.nan, 30, np.nan, 22],
        'Salary': [50000, 60000, np.nan, np.nan, 45000]}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

# Handling missing values

# 1. Remove rows with missing values
df_dropna = df.dropna()

# 2. Fill missing values with a specific value 
df_fillna_mean = df.fillna(df.mean())

# 3. Fill missing values using forward fill
df_ffill = df.ffill()

# 4. Fill missing values using backward fill
df_bfill = df.bfill()

# Display the modified datasets
print("\nAfter Removing Rows with Missing Values:")
print(df_dropna)

print("\nAfter Filling Missing Values with Mean:")
print(df_fillna_mean)

print("\nAfter Forward Fill:")
print(df_ffill)

print("\nAfter Backward Fill:")
print(df_bfill)


# In[ ]:




