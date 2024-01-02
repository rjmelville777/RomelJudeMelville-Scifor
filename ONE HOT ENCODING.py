#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.preprocessing import OneHotEncoder
import numpy as np

colors = ['red', 'blue', 'green']
encoded_colors = encoded_colors.reshape(-1, 1)
onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoded_colors = onehot_encoder.fit_transform(encoded_colors)

print("Original colors:", colors)
print("One-hot encoded colors:\n", onehot_encoded_colors)


# In[ ]:




