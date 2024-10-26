#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from statistics import harmonic_mean
from langdetect import detect
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity


# In[16]:


df = pd.read_csv('coursea_data.csv')  # Replace with the actual path if it's in a subfolder

# Drop the irrelevant columns
df.drop(['Unnamed: 0', 'course_organization'], axis=1, inplace=True)

# Display the DataFrame to check if it's loaded correctly
df.head()


# In[17]:


df


# In[18]:


df.course_students_enrolled.apply(lambda count : count[-1]).value_counts()


# In[19]:


df = df[df.course_students_enrolled.str.endswith('k')]


# In[20]:


df['course_students_enrolled'] = df['course_students_enrolled'].apply(lambda enrolled : eval(enrolled[:-1]) * 1000)
df


# In[22]:


minmax_scaler = MinMaxScaler()
scaled_ratings = minmax_scaler.fit_transform(df[['course_rating','course_students_enrolled']])


# In[23]:


df['course_rating'] = scaled_ratings[:,0]
df['course_students_enrolled'] = scaled_ratings[:,1]
df['overall_rating'] = df[['course_rating','course_students_enrolled']].apply(lambda row : harmonic_mean(row), axis=1)


# In[24]:


df


# In[25]:


df = df[df.course_title.apply(lambda title : detect(title) == 'en')]


# In[26]:


vectorizer = TfidfVectorizer(stop_words='english')
vectors = vectorizer.fit_transform(df.course_title)


# In[27]:


def recommend_by_course_title (title, recomm_count=10) : 
    title_vector = vectorizer.transform([title])
    cosine_sim = cosine_similarity(vectors, title_vector)
    idx = np.argsort(np.array(cosine_sim[:,0]))[-recomm_count:]
    sdf = df.iloc[idx].sort_values(by='overall_rating', ascending=False)
    return sdf


# In[28]:


recommend_by_course_title('A Crash Course in Data Science')


# In[29]:


recommend_by_course_title('machine learning')


# In[30]:


recommend_by_course_title('english')


# In[ ]:




