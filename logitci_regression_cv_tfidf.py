# Importing the libraries
import pandas as pd

nrows = 100000

# Importing the dataset
yelp_text_full = pd.read_csv('./yelp_stemmed_labelled.csv')
yelp_text = yelp_text_full.sample(n=nrows)
del(yelp_text_full)


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
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


tfidf = TfidfVectorizer()
cv = CountVectorizer(max_features=1500)

# Hyperparameters grid
param_grid = [{
               'clf__penalty': ['l2'],
               'clf__C': [1.0, 10.0, 100.0],
               'clf__solver' : ['newton-cg', 'sag'],
               'clf__multi_class' : ['ovr', 'multinomial', 'auto'],
               'clf__fit_intercept': [False, True]
              }
              ]

# Pipeline for CountVectorizer
lr_cv = Pipeline([('vect', cv),
                     ('clf',
                      LogisticRegression(verbose=0))])

# Pipeline for TfidfVectorizer
lr_tfidf = Pipeline([
                     ('tfidf', tfidf),
                     ('clf', LogisticRegression(verbose=0))])

pipelines = [{'cv' : lr_cv}, {'tfidf': lr_tfidf}]

results = {}

for index in range(len(pipelines)):
    for key in pipelines[index]:
    
        gs_lr_tfidf = GridSearchCV(pipelines[index][key],
                                   param_grid,
                                   scoring='accuracy',
                                   cv=10,
                                   verbose=5, 
                                   return_train_score=True)
        
        gs_lr_tfidf.fit(X_train, y_train)
          
        print('Best paramenter set: %s ' % gs_lr_tfidf.best_params_)
        
        print('CV accuracy: %.3f' %gs_lr_tfidf.best_score_)
        
        clf = gs_lr_tfidf.best_estimator_
        print('Test accuracy: %.3f' %clf.score(X_test, y_test))
        
        df_name = next(iter(pipelines[index]))
        
        
        
        results['{0}'.format(df_name)] = pd.DataFrame(gs_lr_tfidf.best_params_, index=[0]) 
        results['{0}'.format(df_name)]['Vectorizer'] = df_name
                
                
        results['{0}'.format(df_name)]['best_score'] = gs_lr_tfidf.best_score_
        results['{0}'.format(df_name)]['test_accuracy'] = clf.score(X_test, y_test)
        
        


import airtable

api_key = 'keyzuMSPCp9CJMHKX'

airtable = airtable.Airtable('appaqpUirAcrQV4GP', 'Logistic regression 100k samples', api_key=api_key)


for key, value in results.items():
    airtable.insert({'Name' : 'Logistic Regression'
            'Vectorizer': str(value['Vectorizer'].iloc[0]), 
                         'clf__C': str(value['clf__C'].iloc[0]),
                         'clf__penalty': str(value['clf__penalty'].iloc[0]),
                         'clf__solver': str(value['clf__solver'].iloc[0]),
                         'clf__multi_class': str(value['clf__multi_class'].iloc[0]),
                         'clf__fit_intercept':str(value['clf__fit_intercept'].iloc[0]),
                         'best_score': str(value['best_score'].iloc[0]),
                         'test_accuracy': str(value['test_accuracy'].iloc[0])
                         })
    
