#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 


# In[2]:


path = r'C:/Users/Yagna/Desktop/yagnaaa/projects/Linear Regression'
dataframe = pd.read_csv(path + '/insurance.csv')
print(dataframe.shape)


# In[3]:


sns.lmplot(x = 'bmi', y = 'charges' , data = dataframe , aspect = 2 , height = 3 ) 


# In[4]:


plt.figure(figsize= (12,4))
sns.heatmap(dataframe.isnull() , cbar = False , cmap = 'viridis' , yticklabels = False )
plt.title("To check missing values in dataset")


# In[5]:


relation = dataframe.corr()
sns.heatmap(relation , cmap= 'Wistia' , annot= True)


# In[7]:


categorical_col = ['sex' , 'children', 'region' , 'smoker']
encoding = pd.get_dummies( data = dataframe , prefix = 'OHE' , prefix_sep = '_' , columns = categorical_col, drop_first = True , dtype = 'int8')


# In[8]:


print('Columns in original data frame:\n',dataframe.columns.values)
print('\nNumber of rows and columns in the dataset:',dataframe.shape)
print('\nColumns in data frame after encoding dummy variable:\n',encoding.columns.values)
print('\nNumber of rows and columns in the dataset:',encoding.shape)


# In[ ]:




