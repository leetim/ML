#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import nltk
import nltk.tokenize as tok
import numpy as np
import sklearn as sk
import gensim as gen
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')


# In[7]:


data = pd.read_csv("news-train.csv")
test = pd.read_csv("test-full.csv")
print(data.keys())


# In[9]:


mapl = lambda f, x: list(map(f, x))
filterl = lambda f, x: list(filter(f, x))
lm = WordNetLemmatizer()
stop_wrods = set(stopwords.words('english'))

# tokens = mapl(tok.wordpunct_tokenize, data.HEADER.get_values())
# tokens = mapl(lambda x: mapl(lm.lemmatize, x), tokens)
# tokens = mapl(lambda x: filterl(lambda y: not (y in stop_wrods or y in string.punctuation), x), tokens)
# print(tokens[0])


# In[10]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def tokenize(s):
    X = mapl(lm.lemmatize, tok.wordpunct_tokenize(s))
    return filterl(lambda y: not (y in stop_wrods or y in string.punctuation), X)

vectorizer = TfidfVectorizer(analyzer="word", tokenizer=tokenize)
vect_data = vectorizer.fit_transform(data.HEADER.get_values())
vect_data.shape


# In[38]:


from sklearn.model_selection import KFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# model = RandomForestClassifier(n_estimators=50)
model = SVC(degree=2, verbose=True)
# model = KNeighborsClassifier(metric=cosine_similarity, n_jobs=3, n_neighbors=10)
params = {"C": range(1, 10), "kernel":["linear", "poly", "rbf", "sigmoid"]}
kfold = KFold(shuffle=True, n_splits=5)
vect_data[:10]
# model.fit(vect_data[100:], data.CAT[100:])
clf = GridSearchCV(param_grid=params, estimator=model, cv=kfold, n_jobs=3, scoring="accuracy", verbose=True)
clf.fit(vect_data[:20000], data.CAT[:20000])
# len(list(filter(lambda x: x, model.predict(vect_data[:100]) == data.CAT[:100])))
# vect_data.shape


# In[ ]:


# model2 = SVC(degree = 2, C = 2, kernel="linear", verbose=True)
# model2.fit(vect_data[:10000], data.CAT[:10000])


# In[ ]:




