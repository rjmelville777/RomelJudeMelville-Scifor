#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv("titanic.csv")
df.head()


# In[3]:


df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns',inplace=True)
df.head()


# In[4]:


inputs = df.drop('Survived',axis='columns')
target = df.Survived


# In[5]:


#inputs.Sex = inputs.Sex.map({'male': 1, 'female': 2})
dummies = pd.get_dummies(inputs.Sex)
dummies.head(3)


# In[6]:


inputs = pd.concat([inputs,dummies],axis='columns')
inputs.head(3)


# In[7]:


inputs.drop(['Sex','male'],axis='columns',inplace=True)
inputs.head(3)


# In[8]:


inputs.columns[inputs.isna().any()]


# In[9]:


inputs.Age[:10]


# In[10]:


inputs.Age = inputs.Age.fillna(inputs.Age.mean())
inputs.head()


# In[11]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(inputs,target,test_size=0.3)


# In[12]:


from sklearn.naive_bayes import GaussianNB
model = GaussianNB()


# In[13]:


model.fit(X_train,y_train)


# In[14]:


model.score(X_test,y_test)


# In[15]:


X_test[0:10]


# In[16]:


y_test[0:10]


# In[17]:


model.predict(X_test[0:10])


# In[18]:


model.predict_proba(X_test[:10])


# In[19]:


from sklearn.model_selection import cross_val_score
cross_val_score(GaussianNB(),X_train, y_train, cv=5)


# In[ ]:





# In[ ]:




