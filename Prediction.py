#!/usr/bin/env python
# coding: utf-8

# ## Name : Yagna Thakkar 

# In[35]:


# importing required libraries 

import pandas as pd
import numpy as np  
import seaborn as sns
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')


# In[36]:


# reading data from dataset URL 
data = pd.read_csv('http://bit.ly/w-data')
print("Data Uploaded!")
data.describe()


# In[38]:


# giving data information like dttypes , size , memory usage , non-null values and count. 
data.info()


# In[39]:


# plotting a simple plot between hours and percentage acquired. 
data.plot(x='Hours', y='Scores', style='*')  
plt.title('Hours v Percentage')  
plt.xlabel('Hours')  
plt.ylabel('Percentage')  
plt.show()


# In[40]:


# plotting sns lmplot for data 
# lmplot gives an optional overlaid regression line. 
# useful for comparing numerical values of the data 
sns.lmplot(x='Hours',y='Scores',data=data)


# In[43]:


# loading X with data of coloumn 1 (hours)
# loading Y with data of coloumn 2 (percentage) 
X = data.iloc[:, :-1].values  
Y = data.iloc[:, 1].values  
print(X)
print(Y)


# In[44]:


# giving a correaltion plot of data. 
# positive correlation indicates that if one value of data increases , the other value also increases. 
relation = data.corr()
sns.heatmap(relation , cmap= 'Wistia' , annot= True)


# In[16]:


# Splitting the data
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, Y, 
                            test_size=0.2, random_state=0) 


# In[45]:


#Training the data using model.fit() 
from sklearn.linear_model import LinearRegression
model = LinearRegression() 
model.fit(X_train, y_train )


# In[46]:


# finding the coefficient of our model 
model.coef_ 


# In[47]:


# predicting our test dataframes. 
predictions = model.predict(X_test)
plt.scatter(y_test,predictions)
plt.xlabel('y-test')
plt.ylabel('predictions')


# In[48]:


# finding MSE and MAE of our model. 
from sklearn import metrics
print('Mean squared error value  :'," ", metrics.mean_squared_error(y_test,predictions))
print('Mean absolute error value  :'," ", metrics.mean_absolute_error(y_test,predictions))


# In[49]:


# Comparing Actual values vs Predicted values
dataframe = pd.DataFrame({'Actual Value': y_test, 'Predicted Value': predictions})  
dataframe 


# In[50]:


# testing with a demo value 
hours = 2.5
hours = np.array(hours).reshape(-1, 1 )
pred1 = model.predict(hours)
print("Predicted Value : " , pred1 )


# In[ ]:




