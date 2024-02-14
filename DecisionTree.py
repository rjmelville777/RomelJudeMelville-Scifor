#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv("titanic.csv")
df.head()


# In[3]:


df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns',inplace=True)


# In[4]:


df.head()


# In[5]:


inputs = df.drop('Survived',axis='columns')
target = df.Survived


# In[6]:


inputs.Sex = inputs.Sex.map({'male': 1, 'female': 2})


# In[7]:


inputs.Age[:10]


# In[8]:


inputs.Age = inputs.Age.fillna(inputs.Age.mean())


# In[9]:


inputs.head()


# In[10]:


from sklearn.model_selection import train_test_split


# In[11]:


X_train, X_test, y_train, y_test = train_test_split(inputs,target,test_size=0.2)


# In[12]:


len(X_train)


# In[13]:


len(X_test)


# In[14]:


from sklearn import tree
model = tree.DecisionTreeClassifier()


# In[15]:


model.fit(X_train,y_train)


# In[16]:


model.score(X_test,y_test)


# In[ ]:





# In[ ]:




