#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install chart_studio')


# # Importing the libraries

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Load the training dataset:

# In[3]:


df=pd.read_csv("kerala.csv")


# In[4]:


df.head(10)


# In[5]:


df.tail(10)


# In[6]:


df.shape


# In[7]:


df.size


# In[8]:


df.info()


# In[9]:


df.describe()


# In[10]:


df["SUBDIVISION"].dtype


# In[11]:


df["YEAR"].dtype


# In[12]:


df["JAN"].dtype


# # Null value detection:

# In[13]:


df.isnull().sum()


# In[14]:


df.FLOODS[df.FLOODS=='YES']=1
df.FLOODS[df.FLOODS=='NO']=0


# In[15]:


df


# # importing matplotlib

# In[16]:


import matplotlib.pyplot as plt


# In[17]:


f=df[['MAR','APR','MAY','JUN']]
g=df[['JUL','AUG','SEP','OCT']]
h=df[['JAN','FEB','NOV','DEC']]


# In[18]:


f.hist()


# In[19]:


g.hist()


# In[20]:


h.hist()


# # creating the model

# In[21]:


from sklearn.linear_model import LogisticRegression


# In[22]:


from sklearn.model_selection import train_test_split


# In[23]:


x=df.iloc[:,1:14]
y=df.iloc[:,-1]


# In[24]:


ml=LogisticRegression()


# In[25]:


ml.fit(x_train,y_train)


# # creating x and y values

# In[26]:


x=df.iloc[:,1:14]
y=df.iloc[:,-1]
y=y.astype(int)


# In[27]:


x_train,x_test,y_train,y_test=train_test_split(x , y , test_size=0.2, random_state=0)
x_train


# # creating Logistic Regression and Accuracy

# In[28]:


ml=LogisticRegression()


# In[29]:


ml.fit(x_train,y_train)


# In[30]:


y1=ml.predict(x_test)


# In[31]:


y1


# In[32]:


y_test.values


# In[33]:


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

ml = LogisticRegression()
lr_clf = ml.fit(x_train,y_train)

lr_accuracy = cross_val_score(lr_clf,x_test,y_test,cv=3,scoring='accuracy',n_jobs=-1)


# In[34]:


lr_accuracy.mean()


# In[35]:


x1=[[2020,14.6,16.6,36.1,110.9,252.6,653.6,687.2,404.7,252.3,270.7,158.6,45.9]]
y1=ml.predict(x1)
y1


# In[36]:


df


# In[37]:


df[["YEAR","FLOODS"]]


# In[56]:


df[["YEAR","JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC","FLOODS"]].sort_values("FLOODS",ascending=True)[0:58]


# In[57]:


X=df[["YEAR","JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC","FLOODS"]].sort_values("FLOODS",ascending=True)[0:58]
print(X.max())
print(X.min())


# In[59]:


df[["YEAR","JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC","FLOODS"]].sort_values("FLOODS",ascending=False)[0:60]


# In[60]:


y=df[["YEAR","JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC","FLOODS"]].sort_values("FLOODS",ascending=False)[0:60]
print(y.max())
print(y.min())


# In[ ]:




