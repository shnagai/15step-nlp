# -*- coding: utf-8 -*-
"""中間層のユニットと学習時のエポックを引数で変更出来るように改良した版

"""

import unicodedata
from os.path import dirname, join, normpath

import MeCab
import neologdn
import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

import sys

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

    def _build_mlp(self, input_dim, hidden_units, output_dim):
        mlp = Sequential()
        mlp.add(Dense(units=hidden_units,
                      input_dim=input_dim,
                      activation='relu'))
        mlp.add(Dense(units=output_dim, activation='softmax'))
        mlp.compile(loss='categorical_crossentropy',
                    optimizer='adam')
        return mlp

    def train(self, texts, labels, units, epochs):  #1
        """学習

        Args:
            texts: 学習データの本文
            labels: 学習データの正解ラベル
            units: ユニット数
            epochs: エポック数

        Returns:
            None: モデルを作成

        """
        vectorizer = TfidfVectorizer(tokenizer=self._tokenize, ngram_range=(1, 2))
        vectorizer.fit(texts)

        # (1)
        # 入力の次元数を取得
        feature_dim = len(vectorizer.get_feature_names())
        # 出力の次元数を取得
        n_labels = max(labels) + 1

        classifier = KerasClassifier(build_fn=self._build_mlp,
                                     input_dim=feature_dim,
                                     hidden_units=units,
                                     output_dim=n_labels)

        pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', classifier),
        ])

        pipeline.fit(texts, labels, classifier__epochs=epochs)

        self.pipeline = pipeline
    def predict(self, texts):
        return self.pipeline.predict(texts)  #2


if __name__ == '__main__':
    BASE_DIR = normpath(dirname(__file__))

    argvs = sys.argv

    units = argvs[0]
    epochs = argvs[1]

    training_data = pd.read_csv(join(BASE_DIR, './assets/dialogue_agent_data/training_data.csv'))

    dialogue_agent = DialogueAgent()
    dialogue_agent.train(training_data['text'], training_data['label'], units, epochs)

    with open(join(BASE_DIR, './assets/dialogue_agent_data/replies.csv')) as f:
        replies = f.read().split('\n')
    input_text = 'お腹空いたな'
    predictions = dialogue_agent.predict([input_text])
    predicted_class_id = predictions[0]

    print(replies[predicted_class_id])
