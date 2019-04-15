# Importing the libraries
import pandas as pd

nrows = 100000

# Importing the dataset
yelp_text_full = pd.read_csv('./yelp_stemmed_labelled.csv')
yelp_text = yelp_text_full.sample(n=nrows)
del (yelp_text_full)

yelp_labels = yelp_text.labels.values
yelp_text = yelp_text.text.values.astype('U')

# Splitting dataset in train / test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(yelp_text,
                                                    yelp_labels,
                                                    test_size=0.2,
                                                    stratify=yelp_labels)

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

tfidf = TfidfVectorizer()
cv = CountVectorizer(max_features=1500)

# Hyperparameters grid
param_grid = [{
    'clf__fit_prior': [True, False],
    'clf__alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
}
]

# Pipeline for CountVectorizer
lr_cv = Pipeline([('vect', cv),
                  ('clf',
                   MultinomialNB())])

# Pipeline for TfidfVectorizer
lr_tfidf = Pipeline([
    ('tfidf', tfidf),
    ('clf', MultinomialNB())])

pipelines = [{'cv': lr_cv}, {'tfidf': lr_tfidf}]

results = {}

for index in range(len(pipelines)):
    for key in pipelines[index]:
        gs_nb_tfidf = GridSearchCV(pipelines[index][key],
                                   param_grid,
                                   scoring='accuracy',
                                   cv=10,
                                   verbose=5,
                                   return_train_score=True)

        gs_nb_tfidf.fit(X_train, y_train)

        print('Best paramenter set: %s ' % gs_nb_tfidf.best_params_)

        print('CV accuracy: %.3f' % gs_nb_tfidf.best_score_)

        clf = gs_nb_tfidf.best_estimator_
        print('Test accuracy: %.3f' % clf.score(X_test, y_test))

        df_name = next(iter(pipelines[index]))

        results['{0}'.format(df_name)] = pd.DataFrame(gs_nb_tfidf.best_params_, index=[0])
        results['{0}'.format(df_name)]['Vectorizer'] = df_name

        results['{0}'.format(df_name)]['best_score'] = gs_nb_tfidf.best_score_
        results['{0}'.format(df_name)]['test_accuracy'] = clf.score(X_test, y_test)

import airtable

api_key = 'keyzuMSPCp9CJMHKX'

airtable = airtable.Airtable('appaqpUirAcrQV4GP', 'Logistic regression 100k samples', api_key=api_key)

for key, value in results.items():
    airtable.insert({'Name': 'Multinomial Naive Bayes',
                     'Vectorizer': str(value['Vectorizer'].iloc[0]),
                     'clf__fit_prior': str(value['clf__fit_prior'].iloc[0]),
                     'clf__alpha': str(value['clf__alpha'].iloc[0])
                     })
