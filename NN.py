
# Importing the libraries
import pandas as pd
import numpy as np

# Importing the dataset and deleting unused columns
dataset = pd.read_csv('yelp_academic_dataset_review.csv')
dataset = dataset.iloc[0:10000, :]

drops = ['business_id', 'cool', 'date', 'funny', 'review_id', 'useful', 'user_id']

dataset = dataset.drop(drops, axis=1)



# Applying preprocessing to classes, if more than 3 stars positive review
dataset[['stars']] = dataset[['stars']] > 3


# Cleaning the texts
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ', dataset['text'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset['stars'].values





# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import History 
from keras import optimizers


model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(1500,)))
model.add(Dense(128,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adamax', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=512)

model.evaluate(X_test, y_test)



