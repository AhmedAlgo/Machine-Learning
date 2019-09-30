#!/usr/bin/env python
# coding: utf-8

# # Typical Data Preparation steps
# 
#  - Getting the necessary python libraries 
#  - Loading the dataset 
#  - Dealing with **Missing values** & **Categorical features** 
#  - Splitting the data into **Training sets** & **Testing sets**
#  - Normalization of features

# ## Importin Python Libraries

# In[28]:


#We will now import some required libraries

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd


# ## Loading the dataset

# In[5]:


#Loading the dataset 

dataset = pd.read_csv('loans.csv') #Store the dataset in a dataframe
print(dataset)

dataset.head() #Show the begining of the file
dataset.tail() #Show the begining of the file


# In[21]:


X = dataset.iloc[:,:-1].values   # [:, :-1] Store all the raws, Store all the columns except the last one

y = dataset.iloc[:,3].values    # [:,3] Store all the raws,  Store colum 3 (Approved)

print (X)
print ()
print (y)


# ## Missing Values
# 
# Rows with missing values can be easily dropped via the dropna method >>> df.dropna(axis=0)
# 
# Similarly, we can drop columns that have at least one NaN in any row by setting the axis argument to 1 >>> df.dropna(axis=1)
# 
# Only drop rows where all columns are NaN >>> df.dropna(how='all’)
# 
# Keep only the rows with at least 2 non-NaN values. >>> df.dropna(thresh=2)
# 
# Only drop rows where NaN appear in specific columns (here: 'C') >>> df.dropna(subset=['C'])
# 
# Note: df is the dataframe

# In[29]:


# Dealing with missing values 

# From the scikit.impute library we first import the SimpleImputer class
from sklearn.impute import SimpleImputer

# Next we define an object of the SimpleImputer class by looking at the docstring (use Shift+Tab)
imputer = SimpleImputer(missing_values=np.nan, strategy='mean') 
imputer.fit(X[:,[1,2]])

X[:,1:3]= imputer.transform(X[:,1:3])


print(X)


# ## Categorical Variables

# In[30]:


#Dealing with categorical variables

#From the scikit.preprocessing library we first import few classes
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 

labelencoder_X = LabelEncoder() 

X[:,0] = labelencoder_X.fit_transform(X[:,0]) 

onehotencoder = OneHotEncoder(categorical_features=[0]) 

X = onehotencoder.fit_transform(X).toarray() 

print(X)


# In[31]:


#Dealing with categorical variables
##From the scikit.preprocessing library we first import few classes

from sklearn.preprocessing import LabelEncoder, OneHotEncoder 

labelencoder_X = LabelEncoder() 
X[:,0] = labelencoder_X.fit_transform(X[:,0]) 
onehotencoder = OneHotEncoder(categorical_features=[0]) 
X = onehotencoder.fit_transform(X).toarray()

#Dependent variable 
labelencoder_y = LabelEncoder() 
y = labelencoder_y.fit_transform(y)

print (y)


# ## Splitting the Data
# 
# - Training Set 
# - Test Set

# In[32]:


# Splitting the data into Training Set and Test Set

from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2) #Test size = 20%, training size = 80% 


# ## Normalization
#  - MinMax Approach
#  - Mean Approach

# In[33]:


#Normalizing the features

from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler() 

X_train = sc_X.fit_transform(X_train) 
X_test = sc_X.transform(X_test)


# In[ ]:




