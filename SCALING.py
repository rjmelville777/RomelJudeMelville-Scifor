#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

data = {
    'x': [10, 20, 30, 40],
    'y': [5, 15, 25, 35],
    'z': [100, 200, 300, 400]
}

df = pd.DataFrame(data)
print("Original Dataset:")
print(df)

# Min-Max Scaling
min_max_scaler = MinMaxScaler()
df_minmax_scaled = pd.DataFrame(min_max_scaler.fit_transform(df), columns=df.columns)

print("Min-Max Scaled Dataset:")
print(df_minmax_scaled)

# Standard Scaling
standard_scaler = StandardScaler()
df_standard_scaled = pd.DataFrame(standard_scaler.fit_transform(df), columns=df.columns)

print("Standard Scaled Dataset:")
print(df_standard_scaled)


# In[ ]:




