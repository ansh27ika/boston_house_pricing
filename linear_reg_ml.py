#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ## Lets load the Boston House Pricing Dataset

# In[ ]:


from sklearn.datasets import load_boston


# In[ ]:


boston = load_boston()


# In[ ]:


boston.keys()


# In[ ]:


# Lets check the description of the dataset
print(boston.DESCR)


# In[ ]:


print(boston.data)


# In[ ]:


print(boston.target)


# In[ ]:


print(boston.feature_names)


# ## Preparing The Dataset

# In[ ]:


dataset = pd.DataFrame(boston.data, columns=boston.feature_names)


# In[ ]:


dataset.head()


# In[ ]:


dataset['Price'] = boston.target


# In[ ]:


dataset.head()


# In[ ]:


dataset.info()


# In[ ]:


# Summarizing The Stats of the data
dataset.describe()


# In[ ]:


# Check the missing Values
dataset.isnull().sum()


# In[ ]:


# EXploratory Data Analysis
# Correlation
dataset.corr()


# In[ ]:


sns.pairplot(dataset)


# ## Analyzing The Correlated Features

# In[ ]:


dataset.corr()


# In[ ]:


plt.scatter(dataset['CRIM'], dataset['Price'])
plt.xlabel("Crime Rate")
plt.ylabel("Price")


# In[ ]:


plt.scatter(dataset['RM'], dataset['Price'])
plt.xlabel("RM")
plt.ylabel("Price")


# In[ ]:


sns.regplot(x="RM", y="Price", data=dataset)


# In[ ]:


sns.regplot(x="LSTAT", y="Price", data=dataset)


# In[ ]:


sns.regplot(x="CHAS", y="Price", data=dataset)


# In[ ]:


sns.regplot(x="PTRATIO", y="Price", data=dataset)


# In[ ]:


# Independent and Dependent features

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]


# In[ ]:


X.head()


# In[ ]:


# y


# In[ ]:


# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)


# In[ ]:


# X_train


# In[ ]:


# X_test


# In[ ]:


# Standardize the dataset
scaler = StandardScaler()


# In[ ]:


X_train = scaler.fit_transform(X_train)


# In[ ]:


X_test = scaler.transform(X_test)


# In[ ]:


pickle.dump(scaler, open('scaling.pkl', 'wb'))


# In[ ]:


# X_train


# In[ ]:


# X_test


# ## Model Training

# In[ ]:


# In[ ]:


regression = LinearRegression()


# In[ ]:


regression.fit(X_train, y_train)


# In[ ]:


# print the coefficients and the intercept
print(regression.coef_)


# In[ ]:


print(regression.intercept_)


# In[ ]:


# on which parameters the model has been trained
regression.get_params()


# In[ ]:


# Prediction With Test Data
reg_pred = regression.predict(X_test)


# In[ ]:


# reg_pred


# ## Assumptions

# In[ ]:


# plot a scatter plot for the prediction
plt.scatter(y_test, reg_pred)


# In[ ]:


# Residuals
residuals = y_test-reg_pred


# In[ ]:


# residuals


# In[ ]:


# Plot this residuals

sns.displot(residuals, kind="kde")


# In[ ]:


# Scatter plot with respect to prediction and residuals
# uniform distribution
plt.scatter(reg_pred, residuals)


# In[ ]:


print(mean_absolute_error(y_test, reg_pred))
print(mean_squared_error(y_test, reg_pred))
print(np.sqrt(mean_squared_error(y_test, reg_pred)))


# ## R square and adjusted R square

#
# Formula
#
# **R^2 = 1 - SSR/SST**
#
#
# R^2	=	coefficient of determination
# SSR	=	sum of squares of residuals
# SST	=	total sum of squares
#

# In[ ]:


score = r2_score(y_test, reg_pred)
print(score)


# In[ ]:


# **Adjusted R2 = 1 – [(1-R2)*(n-1)/(n-k-1)]**
#
# where:
#
# R2: The R2 of the model
# n: The number of observations
# k: The number of predictor variables

# In[ ]:


# display adjusted R-squared
# 1 - (1-score)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)


# ## New Data Prediction

# In[ ]:


boston.data[0].reshape(1, -1)


# In[ ]:


# transformation of new data
scaler.transform(boston.data[0].reshape(1, -1))


# In[ ]:


regression.predict(scaler.transform(boston.data[0].reshape(1, -1)))


# ## Pickling The Model file For Deployment

# In[ ]:


# In[ ]:


pickle.dump(regression, open('regmodel.pkl', 'wb'))


# In[ ]:


pickled_model = pickle.load(open('regmodel.pkl', 'rb'))


# In[ ]:


# Prediction
pickled_model.predict(scaler.transform(boston.data[0].reshape(1, -1)))


# In[ ]:
