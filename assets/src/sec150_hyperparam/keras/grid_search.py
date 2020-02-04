from os.path import dirname, join, normpath

import pandas as pd
from keras import backend as K
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD, Adadelta, Adagrad, Adam
from keras.wrappers.scikit_learn import KerasClassifier
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
def build_model(input_dim, output_dim,  # <1>
                optimizer_class,
                learning_rate,
                dropout=0):
    if K.backend() == 'tensorflow':  # <2>
        K.clear_session()

    mlp = Sequential()
    mlp.add(Dense(units=32, input_dim=input_dim, activation='relu'))
    if dropout:
        mlp.add(Dropout(dropout))
    mlp.add(Dense(units=output_dim, activation='softmax'))

    optimizer = optimizer_class(lr=learning_rate)

    mlp.compile(loss='sparse_categorical_crossentropy',
                metrics=['accuracy'],
                optimizer=optimizer)

    return mlp


parameters = {'optimizer_class': [SGD, Adagrad, Adadelta, Adam],  # <3>
              'learning_rate': [0.1, 0.01, 0.001, 0.0001, 0.00001],
              'dropout': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
              'epochs': [10, 50, 100, 200],
              'batch_size': [16, 32, 64]}

feature_dim = train_vectors.shape[1]
n_labels = max(train_labels) + 1

model = KerasClassifier(build_fn=build_model,  # <4>
                        input_dim=feature_dim,
                        output_dim=n_labels,
                        verbose=0)
gridsearch = GridSearchCV(estimator=model, param_grid=parameters)  # <5>
gridsearch.fit(train_vectors, train_labels)

print('Best params are: {}'.format(gridsearch.best_params_))

# Load test data
test_data = pd.read_csv(join(BASE_DIR, './test_data.csv'))
test_texts = test_data['text']
test_labels = test_data['label']

# Classification with the best parameters
test_vectors = vectorizer.transform(test_texts)
predictions = gridsearch.predict(test_vectors)
