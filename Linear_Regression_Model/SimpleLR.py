#!/usr/bin/env python
# coding: utf-8

# ## Data Preparation

# In[1]:


#import the required libraries
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd

#Loading the dataset 
dataset = pd.read_csv('gm.csv') #Store the dataset in a dataframe

X = dataset.iloc[:,:-1].values   # [:, :-1] Store all the raws, Store all the columns except the last one
y = dataset.iloc[:,1].values    # [:,1] Store all the raws,  Store colum 1

# Splitting the data into Training Set and Test Set
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3) #Test size = 30%, training size = 70% 

#Normalizing the features
#from sklearn.preprocessing import StandardScaler 
#sc_X = StandardScaler() 
#X_train = sc_X.fit_transform(X_train) 
#X_test = sc_X.transform(X_test)


# ## Building a Linear Regression Model

# In[2]:


#Fitting Linear Regression to Training Set 
from sklearn.linear_model import LinearRegression 

lrObj = LinearRegression() 
lrObj.fit(X_train, y_train)


# In[4]:


#Prediction on the Test Set 
y_pred = lrObj.predict(X_test)


# In[5]:


#We can compare the predicted values with the actual values 
print(y_test) 


# In[6]:


print(y_pred)


# ## Visual Exploration

# In[7]:


#Visual Exploration of Training Set 

plt.scatter(X_train,y_train,color='red') 
plt.plot(X_train, lrObj.predict(X_train), color='blue') 

plt.title('List Price vs Best Price on Training Set') 
plt.xlabel('List Price') 
plt.ylabel('Best Price') 

plt.show()


# In[8]:


#Visual Exploration of Testing Set 

plt.scatter(X_test,y_test,color='red') 
plt.plot(X_train, lrObj.predict(X_train), color='blue') 

plt.title('List Price vs Best Price on Testing Set') 
plt.xlabel('List Price') 
plt.ylabel('Best Price') 

plt.show()


# In[ ]:




