#!/usr/bin/env python
# coding: utf-8

# # Sentiment Analysis on Musical Instruments Reviews

# This project is based on sentiments analysis on the data extracted from Kaggle. The data relies on the reviews of musical instruments. I performed several functions to analyse the data. The list of libraries and programs are given below:

# In[11]:


#Data
import numpy as np
import pandas as pd
#NLTK
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
#Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud  
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
# Data Modeling
from sklearn.pipeline import Pipeline 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import GridSearchCV
# NGrams
from nltk.util import ngrams
from collections import Counter
import os
import warnings


# In[12]:


data= pd.read_csv('Musical_instruments_reviews.csv') 


# In[13]:


data.head()


# In[14]:


data.info()


# In[15]:


data.shape


# In[16]:


data = data.iloc[:,1:]


# In[17]:


data.head()


# In[18]:


data.describe()


# In[19]:


data['summary'].value_counts()


# In[20]:


data.isnull().sum()


# In[21]:


print("Number of duplicates: " + str(data.duplicated().sum()))


# In[22]:


print('### Categorical features ###','\n')
data.describe(include=['O'])


# In[23]:


print('### Numerical features ###','\n')
data.describe(exclude=['O'])


# In[25]:


data.nunique()


# In[26]:


data['overall'].value_counts()


# In[28]:


print("Maximum: " + str(data['overall'].max()))
print("Minimum: " + str(data['overall'].min()))


# In[33]:


data['overall'].value_counts()


# In[34]:


data = data[['overall', 'summary']].copy() #create the dataframe only with overall and summary
data.head()


# In[35]:


def conv(row):                          # This function will return the value with sentiments of positive and negative 
    
    if row['overall'] < 2.0:
        val = 'Negative'
    elif row['overall'] > 3.0:
        val = 'Positive'
    else: 
        val = 'Neutral'
    
    return val


# In[36]:


data['overall'] = data.apply(conv, axis=1) 
data


# In[31]:


import seaborn as sns
sns.countplot(x=data['overall'],data=data)  #Find the value of the data with seaborn


# In[37]:


data['overall'].value_counts()


# In[75]:


from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(df,test_size=0.2,random_state=123)


# In[76]:


X_train = train_df['overall']
X_test = test_df['overall']
y_train = train_df['overall']
y_test = test_df['overall']


# In[77]:


# Build a pipeline to find out optimized parameters of TfidfVectorizer and Logistic Regression.
pipeline = Pipeline([
    ('tfidf_vect', TfidfVectorizer(stop_words='english')),
    ('lr_clf', LogisticRegression(solver='liblinear'))
])

params = {'tfidf_vect__ngram_range': [(1,1), (1,2), (1,3)],
          'tfidf_vect__max_df': [0.25, 0.50, 0.75],
          'lr_clf__C': [1, 10, 20]}

grid_cv_pipe = GridSearchCV(pipeline, param_grid=params, cv=3, scoring='accuracy', verbose=1)
grid_cv_pipe.fit(X_train, y_train)
print('Optimized Hyperparameters: ', grid_cv_pipe.best_params_)

pred = grid_cv_pipe.predict(X_test)
print('Optimized Accuracy Score: {0: .3f}'.format(accuracy_score(y_test, pred)))


# In[78]:


corpus = df['summary']

# Concatenate the text into a single string
text = ' '.join(corpus)

# Remove symbols and special characters using regex
text = re.sub(r'[^\w\s]', '', text)

# Tokenize the cleaned text into words
words = nltk.word_tokenize(text)

# Create 5-grams
n = 5
five_grams = [' '.join(gram) for gram in ngrams(words, n)]


# In[79]:


gram_counts = Counter(five_grams)

# Get the most common 5-grams
top_n = 10  # Change this value to display the top N 5-grams
top_grams = gram_counts.most_common(top_n)

# Extract the 5-grams and their frequencies
top_gram_names = [gram for gram, count in top_grams]
top_gram_counts = [count for gram, count in top_grams]

# Plot the top 5-grams and their frequencies
plt.figure(figsize=(12, 6))
plt.barh(top_gram_names, top_gram_counts)
plt.xlabel('Frequency')
plt.ylabel('5-Gram')
plt.title(f'Top {top_n} 5-Grams by Frequency')
plt.gca().invert_yaxis()  # Invert the y-axis to display the most frequent at the top
plt.show()

