import math
from os.path import dirname, join, normpath

import numpy as np
import pandas as pd
from hyperopt import fmin, hp, tpe
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD, Adadelta, Adagrad, Adam
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
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

feature_dim = train_vectors.shape[1]
n_labels = max(train_labels) + 1

tr_labels, val_labels, tr_vectors, val_vectors =\
    train_test_split(train_labels, train_vectors, random_state=42)

input_dim = feature_dim
output_dim = n_labels

# Hyperparameter search

train_epochs = 200


def build_mlp_model(hidden_units, dropout, optimizer):  # <1>
    mlp = Sequential()
    mlp.add(Dense(units=hidden_units, input_dim=input_dim, activation='relu'))
    if dropout:
        mlp.add(Dropout(dropout))
    mlp.add(Dense(units=output_dim, activation='softmax'))

    mlp.compile(loss='sparse_categorical_crossentropy',
                metrics=['accuracy'],
                optimizer=optimizer)

    return mlp


def objective(args):  # <2>
    if K.backend() == 'tensorflow':  # <3>
        K.clear_session()

    hidden_units = int(args['hidden_units'])  # <4>
    dropout = args['dropout']

    optimizer_class, optimizer_args = args['optimizer']
    optimizer = optimizer_class(**optimizer_args)

    mlp = build_mlp_model(hidden_units, dropout, optimizer)

    batch_size = max(int(args['batch_size']), 1)  # <5>
    history = mlp.fit(tr_vectors,
                      tr_labels,
                      epochs=train_epochs,
                      batch_size=batch_size,
                      callbacks=[
                          EarlyStopping(min_delta=0.0, patience=3)],  # <6>
                      validation_data=(val_vectors, val_labels))
    if len(history.history['val_loss']) == train_epochs:
        print('[WARNING] Early stopping did not work')

    val_pred = np.argmax(mlp.predict(val_vectors), axis=1)

    accuracy = accuracy_score(val_pred, val_labels)
    return -accuracy


# <7>
space = {'optimizer': hp.choice('optimizer', [
            (SGD, {'lr': hp.loguniform('lr_sgd',
                                       math.log(1e-6),
                                       math.log(1)),
                   'momentum': hp.uniform('momentum',
                                          0,
                                          1)}),
            (Adagrad, {'lr': hp.loguniform('lr_adagrad',
                                           math.log(1e-6),
                                           math.log(1))}),
            (Adadelta, {'lr': hp.loguniform('lr_adadelta',
                                            math.log(1e-6),
                                            math.log(1))}),
            (Adam, {'lr': hp.loguniform('lr_adam',
                                        math.log(1e-6),
                                        math.log(1))})]),
         'hidden_units': hp.qloguniform('hidden_units',
                                        math.log(32),
                                        math.log(256),
                                        1),
         'batch_size': hp.qloguniform('batch_size',
                                      math.log(1),
                                      math.log(256),
                                      1),
         'dropout': hp.uniform('dropout', 0, 0.5)}

best = fmin(objective, space, algo=tpe.suggest, max_evals=100)  # <8>

print('Best params are: {}'.format(best))  # <9>

# Create a model with the best params and train it
# <10>
optimizer_choices = [node.pos_args[0].obj
                     for node in space['optimizer'].pos_args[1:]]
BestOptimizer = optimizer_choices[best['optimizer']]
optimizer_args = {}
if BestOptimizer == SGD:
    optimizer_args['lr'] = best['lr_sgd']
    optimizer_args['momentum'] = best['momentum']
elif BestOptimizer == Adagrad:
    optimizer_args['lr'] = best['lr_adagrad']
elif BestOptimizer == Adadelta:
    optimizer_args['lr'] = best['lr_adadelta']
elif BestOptimizer == Adam:
    optimizer_args['lr'] = best['lr_adam']

optimizer = BestOptimizer(**optimizer_args)

hidden_units = int(best['hidden_units'])
dropout = best['dropout']

mlp = build_mlp_model(hidden_units, dropout, optimizer)

batch_size = max(int(best['batch_size']), 1)
mlp.fit(train_vectors,
        train_labels,
        epochs=train_epochs,
        batch_size=batch_size,
        callbacks=[EarlyStopping(min_delta=0.0, patience=1)])

# Load test data
test_data = pd.read_csv(join(BASE_DIR, './test_data.csv'))
test_texts = test_data['text']
test_labels = test_data['label']

# Classification with the best parameters
test_vectors = vectorizer.transform(test_texts)
test_preds = np.argmax(mlp.predict(test_vectors), axis=1)

print(accuracy_score(test_preds, test_data['label']))
