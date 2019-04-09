#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 08:50:40 2019

@author: robertosannazzaro
"""

import pandas as pd

yelp = pd.read_csv('./yelp_dataset/yelp_academic_dataset_review.csv')
yelp = yelp.dropna()
#yelp = yelp.drop('Unnamed: 0', axis=1)
yelp.head()

print('Dataset loaded')


from nltk.stem.porter import PorterStemmer
import re

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
for date, row in yelp.text.iteritems():
    print(date)
    review = re.sub('[^a-zA-Z]', ' ', row)
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    print(review)
    
print('Stemming completed!')

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features = 1500)
Xcv = cv.fit_transform(corpus).toarray()
ycv = yelp['positive']

print('CV completed!')

cvDF = pd.DataFrame(Xcv)
cvDF.to_csv('cvDF.csv')

print('Converted to csv!')

from sklearn.feature_extraction.text import TfidfVectorizer

tf = TfidfVectorizer()

Xtf = tf.fit_transform(corpus).toarray()
ytf = yelp['positive']

print('TF Completed!')

cvTF = pd.DataFrame(Xtf)
cvTF.to_csv('cvTF.csv')

print('Converted to csv!')










