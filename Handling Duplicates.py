#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[12]:


import pandas as pd
data = {
    'Product': ['Laptop', 'Mouse', 'Keyboard', 'Laptop', 'Mouse', 'Monitor'],
    'Price': [1000, 20, 50, 1000, 25, 300],
    'Quantity': [5, 10, 8, 5, 12, 2]
}

df = pd.DataFrame(data)
print("Original DataFrame with Duplicates:")
print(df)


# In[13]:


duplicates = df[df.duplicated(keep=False)]

print("\nDuplicate Rows except first occurrence:")
print(duplicates)


df_no_duplicates = df.drop_duplicates()

print("\nDataFrame without Duplicates:")
print(df_no_duplicates)


# In[ ]:




