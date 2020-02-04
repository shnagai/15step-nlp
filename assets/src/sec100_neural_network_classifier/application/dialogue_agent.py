import unicodedata
from os.path import dirname, join, normpath

import MeCab
import neologdn
import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.feature_extraction.text import TfidfVectorizer


class DialogueAgent:
    def __init__(self):
        self.tagger = MeCab.Tagger()

    def _tokenize(self, text):
        text = unicodedata.normalize('NFKC', text)
        text = neologdn.normalize(text)
        text = text.lower()

        node = self.tagger.parseToNode(text)
        result = []
        while node:
            features = node.feature.split(',')

            if features[0] != 'BOS/EOS':
                if features[0] not in ['助詞', '助動詞']:
                    token = features[6] \
                            if features[6] != '*' \
                            else node.surface
                    result.append(token)

            node = node.next

        return result

    def train(self, texts, labels):
        vectorizer = TfidfVectorizer(tokenizer=self._tokenize,
                                     ngram_range=(1, 2))
        tfidf = vectorizer.fit_transform(texts)

        # <1>
        feature_dim = len(vectorizer.get_feature_names())
        n_labels = max(labels) + 1

        # <2>
        mlp = Sequential()
        mlp.add(Dense(units=32,
                      input_dim=feature_dim,
                      activation='relu'))
        mlp.add(Dense(units=n_labels, activation='softmax'))
        mlp.compile(loss='categorical_crossentropy',
                    optimizer='adam')

        # <3>
        labels_onehot = to_categorical(labels, n_labels)
        mlp.fit(tfidf, labels_onehot, epochs=100)

        self.vectorizer = vectorizer
        self.mlp = mlp

    def predict(self, texts):
        tfidf = self.vectorizer.transform(texts)
        predictions = self.mlp.predict(tfidf)
        predicted_labels = np.argmax(predictions, axis=1)  # <4>
        return predicted_labels


if __name__ == '__main__':
    BASE_DIR = normpath(dirname(__file__))

    training_data = pd.read_csv(join(BASE_DIR, './training_data.csv'))

    dialogue_agent = DialogueAgent()
    dialogue_agent.train(training_data['text'], training_data['label'])

    with open(join(BASE_DIR, './replies.csv')) as f:
        replies = f.read().split('\n')

    input_text = '名前を教えてよ'
    predictions = dialogue_agent.predict([input_text])
    predicted_class_id = predictions[0]

    print(replies[predicted_class_id])
