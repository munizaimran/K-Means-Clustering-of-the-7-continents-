#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
#setting seaborn for data visualization
sns.set()
#sklearn for adv statistical methods
from sklearn.cluster import KMeans


# In[3]:


data = pd.read_csv("5.1 Categorical.csv")
data


# In[5]:


#plotting longitude on x-axis (-180 to 180) and latitude on y-axis (-90 to 90)
plt.scatter(data['Longitude'], data['Latitude'])
plt.xlim(-180, 180)
plt.ylim(-90, 90)
plt.show()


# In[6]:


#pandas data slicing function (slicing the 0th and 3rd column, keeping 1st and 2nd column features)
x = data.iloc[:,1:3]
x


# In[15]:


#KMeans is the method from sklearn. the number in bracket indicates the number of clusters being used.
#the variable kmeans becomes the object
kmeans = KMeans(7)
#using the kmeans object with the 'fit' function implements the actual function of clustering
#we'll be clustering the sliced data (x) 
kmeans.fit(x)


# In[16]:


#array of predicted cluster is created by the "fit_predict" method and is stored in the variable "identified_clusters"
identified_clusters = kmeans.fit_predict(x)
identified_clusters


# In[17]:


#forming a dataset including clustering
#first copy the data
data_with_clusters = data.copy()
# Create a new column and add the identified cluster array in the dataset
data_with_clusters['Cluster'] = identified_clusters
data_with_clusters


# In[18]:


plt.scatter(data_with_clusters['Longitude'],data_with_clusters['Latitude'],c=data_with_clusters['Cluster'],cmap='rainbow')
#c is an argument for color for different clusters
plt.xlim(-180,180)
plt.ylim(-90,90)
plt.xlabel("Latitude")
plt.ylabel("Longitude")
plt.title("Clustering of the 7 Continents")
plt.show()


# In[ ]:




