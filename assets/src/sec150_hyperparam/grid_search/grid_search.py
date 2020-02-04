from os.path import dirname, join, normpath

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV

from tokenizer import tokenize

# Load training data
BASE_DIR = normpath(dirname(__file__))

training_data = pd.read_csv(join(BASE_DIR, './training_data.csv'))
train_texts = training_data['text']
train_labels = training_data['label']

# Feature extraction
vectorizer = TfidfVectorizer(tokenizer=tokenize, ngram_range=(1, 2))
train_vectors = vectorizer.fit_transform(train_texts)

# Grid search
parameters = {  # <1>
    'n_estimators': [10, 20, 30, 40, 50, 100, 200, 300, 400, 500],
    'max_features': ('sqrt', 'log2', None),
}
classifier = RandomForestClassifier()
gridsearch = GridSearchCV(classifier, parameters)  # <2>

gridsearch.fit(train_vectors, train_labels)  # <3>

print('Best params are: {}'.format(gridsearch.best_params_))  # <4>

# Load test data
test_data = pd.read_csv(join(BASE_DIR, './test_data.csv'))
test_texts = test_data['text']
test_labels = test_data['label']

# Classification with the best parameters
test_vectors = vectorizer.transform(test_texts)
predictions = gridsearch.predict(test_vectors)  # <5>
