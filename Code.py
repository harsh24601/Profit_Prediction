#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn








# In[3]:


dataset  = pd.read_csv('C:/Users/harsh/Videos/Data Analyst Skillpath_ Zero to Hero in Excel, SQL & Python/Data files/Part 3 - Regression using Python/50_Startups.csv',header = 0)








# In[4]:


dataset.head()







# In[5]:


dataset.tail()







# In[6]:


dataset.describe()







# In[7]:


print('There are' , dataset.shape[0],'rows and' , dataset.shape[1],'columns in the dataset')







# In[8]:


print('There are' , dataset.duplicated().sum(),'duplicate values in the dataset')







# In[9]:


dataset.isnull().sum()







# In[10]:


dataset.info()








# In[11]:


c=dataset.corr()
c








# In[12]:


sns.heatmap(c,annot=True,cmap='Blues')
plt.show()








# In[13]:


outliers = ['Profit']
plt.rcParams['figure.figsize'] =[8,8]
sns.boxplot(data=dataset[outliers], orient='v', palette = 'Set2' , width =0.7)
plt.title ('Outliers Variables Distribution ')
plt.ylabel('Profit Range')
plt.xlabel('Continuous Variable ')
plt.show()







# In[14]:


sns.distplot(dataset['Profit'], bins=5, kde=True)
plt.show()








# In[15]:


sns.pairplot(dataset)
plt.show()







# In[16]:


x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values







# In[17]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.7 , random_state =0)
x_train







# In[18]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)
('Model has been trained successfully ')








# In[19]:


y_pred = model.predict(x_test)
y_pred











testing_data_model_score = model.score(x_test,y_test)
testing_data_model_score








# In[21]:


df = pd.DataFrame(data={'Predicted value' : y_pred.flatten(), 'Actual value' : y_test.flatten()})
df








# In[22]:


from sklearn.metrics import r2_score
r2_score = r2_score (y_pred,y_test)
print('R2 score of the model is' ,r2_score)








# In[23]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_pred,y_test)
print('Mean squared error of the model is' ,mse)






# In[24]:


import numpy as np
rmse = np.sqrt(mean_squared_error(y_pred,y_test))
print(' Root mean squared error of the model is' ,rmse)








# In[25]:


from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_pred,y_test)
print('Mean absolute error of the model is' ,mse)






