#!/usr/bin/env python
# coding: utf-8

# In[35]:


# import the libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string


# In[36]:


# fake news data dataframe 

df_fake = pd.read_csv("Fake.csv")
df_fake.head()


# In[37]:


# True News data Frame 

df_true = pd.read_csv("True.csv")
df_true.head()


# In[38]:


# adding target column into the fake and True datasets 

# fake new == 0
# Not a fake news == 1

df_fake["class"] = 0
df_fake.head()


# In[39]:


df_true["class"] = 1
df_true.head()


# In[40]:


# checking the shape 
df_fake.shape, df_true.shape


# In[41]:


# extracting 10 rows from fake news data and True news data 
# so we can test our model

df_fake_manual_testing = df_fake.tail(10)

for i in range(23480,23470,-1):
    df_fake.drop([i], axis = 0, inplace = True)

df_fake_manual_testing
#len(df_fake_manual_testing)


# In[42]:


df_true_manual_testing = df_true.tail(10)

for i in range(21416,21406,-1):
    df_true.drop([i], axis = 0, inplace = True)
    
#len(df_true_manual_testing)


# In[43]:


df_fake.shape, df_true.shape


# In[44]:


# adding class 
df_fake_manual_testing["class"] = 0
df_true_manual_testing["class"] = 1


# In[45]:


df_fake_manual_testing


# In[46]:


df_true_manual_testing


# In[47]:


# create manual dataset for testing

df_manual_testing = pd.concat([df_fake_manual_testing,df_true_manual_testing], axis = 0)

df_manual_testing.to_csv("manual_testing.csv" , index = False)


# In[48]:


# Now deal with actual data 

df_merge = pd.concat([df_fake, df_true], axis =0 )
df_merge


# In[49]:


df_merge.columns


# In[50]:


# drop columns 

df = df_merge.drop(["title", "subject","date"], axis = 1)
df.head()


# In[51]:


df.isnull().sum()


# In[52]:


# Randomly shuffling the dataframe 

# The frac keyword argument specifies the fraction of rows to return in the random sample, 
# so frac=1 means return all rows (in random order).


df = df.sample(frac = 1)


# In[53]:


df.head(10)


# In[54]:


# becouse of shuffeling all the data index also shuffle 
# so reset the index

df.reset_index(inplace = True)

df.drop(["index"], axis = 1, inplace = True)


# In[55]:


df.head()


# In[56]:


# check columns

df.columns


# In[57]:


# Creating a function to convert the text in lowercase, 
# remove the extra space, special chr., ulr and links.


def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text


# In[58]:


df["text"] = df["text"].apply(wordopt)


# In[59]:


df


# In[60]:


#  Defining dependent and independent variable as x and y

x = df["text"]
y = df["class"]


# In[61]:


#  Splitting the dataset into training set and testing set. 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25 ,random_state = 0)


# In[62]:



from sklearn.feature_extraction.text import CountVectorizer

vectorization = CountVectorizer()


# In[63]:


xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)


# In[64]:


# 1 - logistic regression

from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()
LR.fit(xv_train,y_train)

pred_lr=LR.predict(xv_test)

LR.score(xv_test, y_test)*100


# In[65]:


#  2 - Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier

RFC = RandomForestClassifier(random_state=0)
RFC.fit(xv_train, y_train)

pred_rfc = RFC.predict(xv_test)

RFC.score(xv_test, y_test)*100


# In[66]:


# 3 - MultinomialNB 

from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(xv_train, y_train)
y_pred = model.predict(xv_train)
model.score(xv_test, y_test)*100


# In[67]:


# Model Testing With Manual Entry


def output_lable(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not a Fake News"


def manual_testing(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    
    new_def_test["text"] = new_def_test["text"].apply(wordopt) 
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    
    
    pred_LR = LR.predict(new_xv_test)  # [0]
    pred_naive_bayes = model.predict(new_xv_test)  # [1]
    pred_RFC = RFC.predict(new_xv_test)   # [1]
    

    return print("\n\nLR Prediction: {} \nNaive Bayes: {} \nRFC Prediction: {}".format(output_lable(pred_LR[0]),
                                                                                     output_lable(pred_naive_bayes[0]),
                                                                                     output_lable(pred_RFC[0])))


# In[68]:


news = str(input())
manual_testing(news)o


# In[ ]:




