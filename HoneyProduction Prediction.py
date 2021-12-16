#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model


# In[7]:


pwd


# In[9]:


df = pd.read_csv("C:\\Users\\mondi\\Downloads\\honeyproduction.csv")


# In[11]:


print(df.head(5))


# In[12]:


prod_per_year = df.groupby('year').totalprod.mean().reset_index()


# In[13]:


X = prod_per_year["year"]
X = X.values.reshape(-1, 1)
y = prod_per_year["totalprod"]


# In[14]:


regr = linear_model.LinearRegression()
regr.fit(X,y)


# In[15]:


print(regr.coef_)
print(regr.intercept_)


# In[16]:


y_predict = regr.predict(X)
print(y_predict)


# In[18]:


plt.scatter(X, y, alpha=0.2)
plt.plot(X, y_predict)
plt.show()


# In[19]:


X_future = np.array(range(2013, 2051))
X_future = X_future.reshape(-1,1)


# In[20]:


print(X_future)
print(df.head(5))


# In[22]:


future_predict = regr.predict(X_future)


# In[26]:


plt.scatter(X, y, alpha=0.2)
plt.plot(X, y_predict)
#till 2012

#after 2012 till 2050
plt.plot(X_future, future_predict)
plt.show()

