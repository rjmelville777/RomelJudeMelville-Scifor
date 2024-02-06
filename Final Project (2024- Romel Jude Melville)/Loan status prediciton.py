#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data = pd.read_csv('loan_prediction.csv')


# In[3]:


data.head()


# In[4]:


data.tail()


# In[5]:


data.shape


# In[6]:


data.info()


# In[7]:


data.isnull().sum()


# In[8]:


data.isnull().sum() * 100 / len(data)


# In[9]:


#Handling missing values
data = data.drop('Loan_ID', axis = 1)


# In[10]:


columns = ['Gender', 'Dependents','LoanAmount','Loan_Amount_Term']


# In[11]:


data = data.dropna(subset=columns)


# In[12]:


data.isnull().sum() * 100 / len(data)


# In[13]:


data['Self_Employed'] = data['Self_Employed'].fillna(data['Self_Employed'].mode()[0])


# In[14]:


data.isnull().sum() * 100 / len(data)


# In[15]:


data['Credit_History'] = data['Credit_History'].fillna(data['Credit_History'].mode()[0])


# In[16]:


data.isnull().sum() * 100 / len(data)


# In[17]:


#Handling Categorical columns
data.sample(10)


# In[18]:


data['Dependents'] = data['Dependents'].replace(to_replace = "3+",value = '4')


# In[19]:


data['Gender'] = data['Gender'].map({'Male':1,'Female':0}).astype('int')
data['Married'] = data['Married'].map({'Yes':1,'No':0}).astype('int')
data['Education'] = data['Education'].map({'Graduate':1,'Not Graduate':0}).astype('int')
data['Self_Employed'] = data['Self_Employed'].map({'Yes':1,'No':0}).astype('int')
data['Property_Area'] = data['Property_Area'].map({'Rural':0,'Urban':1,'Semiurban' : 2}).astype('int')
data['Loan_Status'] = data['Loan_Status'].map({'Y':1,'N':0}).astype('int')


# In[20]:


data.sample(10)


# In[21]:


X = data.drop('Loan_Status',axis = 1)


# In[22]:


y = data['Loan_Status']
y


# In[23]:


#feature scaling
data.head()


# In[24]:


cols = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']


# In[25]:


from sklearn.preprocessing import StandardScaler
st = StandardScaler()
X[cols] = st.fit_transform(X[cols])


# In[26]:


X


# In[27]:


#Splitting data into training and testing sets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import numpy as np


# In[28]:


model_df = {}
def model_val(model,X,y):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20, random_state=42)
    
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    print(f"{model} accuracy is {accuracy_score(y_test,y_pred)}")
    
    score = cross_val_score(model,X,y,cv=5)
    print(f"{model} Avg cross val score is {np.mean(score)}")
    model_df[model] = round(np.mean(score)*100,2)


# In[29]:


#Logistic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model_val(model,X,y)


# In[30]:


#SVC
from sklearn import svm
model = svm.SVC()
model_val(model,X,y)


# In[31]:


#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model_val(model,X,y)


# In[32]:


#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
model =RandomForestClassifier()
model_val(model,X,y)


# In[33]:


#Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
model =GradientBoostingClassifier()
model_val(model,X,y)


# In[34]:


model_df


# In[35]:


#Hyperparamater tuning
from sklearn.model_selection import RandomizedSearchCV


# In[36]:


#Logistic Regression
log_reg_grid = {"C":np.logspace(-4,4,20),"solver":['liblinear']}
lr_log_reg = RandomizedSearchCV(LogisticRegression(),param_distributions = log_reg_grid, n_iter = 20,cv = 5,verbose = True)


# In[37]:


lr_log_reg.fit(X,y)


# In[38]:


lr_log_reg.best_score_


# In[39]:


lr_log_reg.best_params_


# In[40]:


#SVC
svc = {'C':[0.25,0.50,0.75,1],"kernel":["linear"]}


# In[41]:


lr_svc = RandomizedSearchCV(svm.SVC(), param_distributions = svc, cv = 5, n_iter = 20, verbose = True)


# In[42]:


lr_svc.fit(X,y)


# In[43]:


lr_svc.best_score_


# In[44]:


lr_svc.best_params_


# In[45]:


#Random Forest Classifier
RandomForestClassifier()


# In[46]:


rf = {'n_estimators':np.arange(10,1000,10),'max_features':['auto','sqrt'],'max_depth':[None,3,5,10,20,30],
'min_samples_split':[2,5,20,50,100],'min_samples_leaf':[1,2,5,10]}


# In[47]:


lr_rf = RandomizedSearchCV(RandomForestClassifier(), param_distributions = rf,  cv = 5,  n_iter = 20, verbose = True)


# In[48]:


lr_rf.fit(X,y)


# In[49]:


lr_rf.best_score_


# In[50]:


lr_rf.best_params_


# In[51]:


#Deploying the model
X = data.drop('Loan_Status',axis=1)
y = data['Loan_Status']


# In[52]:


rf = RandomForestClassifier(n_estimators = 250,min_samples_split = 5,min_samples_leaf = 2,max_features ='sqrt',max_depth = 3)


# In[53]:


rf.fit(X,y)


# In[54]:


import joblib


# In[55]:


joblib.dump(rf,'loan_status_prediction')


# In[56]:


model = joblib.load('loan_status_prediction')


# In[57]:


import pandas as pd
df = pd.DataFrame({
    'Gender':1,
    'Married':1,
    'Dependents':2,
    'Education':0,
    'Self_Employed':0,
    'ApplicantIncome':2900,
    'CoapplicantIncome':0.0,
    'LoanAmount':45,
    'Loan_Amount_Term':180,
    'Credit_History':0,
    'Property_Area':1
},index=[0])


# In[58]:


df


# In[59]:


result = model.predict(df)


# In[60]:


if result==1:
    print("Loan Approved")
else:
    print("Loan Not Approved")

