from os.path import dirname, join, normpath

import pandas as pd
from hyperopt import fmin, hp, tpe
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from tokenizer import tokenize

# Load training data
BASE_DIR = normpath(dirname(__file__))

training_data = pd.read_csv(join(BASE_DIR, './training_data.csv'))
train_texts = training_data['text']
train_labels = training_data['label']

# Feature extraction
vectorizer = TfidfVectorizer(tokenizer=tokenize, ngram_range=(1, 2))
train_vectors = vectorizer.fit_transform(train_texts)

tr_labels, val_labels, tr_vectors, val_vectors =\
    train_test_split(train_labels, train_vectors, random_state=42)


# Search
def objective(args):  # <1>
    classifier = RandomForestClassifier(n_estimators=int(args['n_estimators']),
                                        max_features=args['max_features'])
    classifier.fit(tr_vectors, tr_labels)
    val_predictions = classifier.predict(val_vectors)
    accuracy = accuracy_score(val_predictions, val_labels)
    return -accuracy


max_features_choices = ('sqrt', 'log2', None)
space = {  # <2>
    'n_estimators': hp.quniform('n_estimators', 10, 500, 10),
    'max_features': hp.choice('max_features', max_features_choices),
}

best = fmin(objective, space, algo=tpe.suggest, max_evals=30)  # <3>

# Create a classifier with the best params and train it
best_classifier = RandomForestClassifier(  # <4>
    n_estimators=int(best['n_estimators']),
    max_features=max_features_choices[best['max_features']])
best_classifier.fit(train_vectors, train_labels)

# flake8: noqa: E402
from sklearn.metrics import accuracy_score

# Load test data
test_data = pd.read_csv(join(BASE_DIR, './test_data.csv'))
test_texts = test_data['text']
test_labels = test_data['label']

# Clasification with the best parameters
test_vectors = vectorizer.transform(test_texts)
predictions = best_classifier.predict(test_vectors)

print(accuracy_score(test_labels, predictions))
