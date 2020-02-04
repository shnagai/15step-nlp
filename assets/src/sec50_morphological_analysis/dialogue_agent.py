from os.path import dirname, join, normpath

import MeCab
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC

MECAB_DIC_DIR = '/usr/lib/mecab/dic/mecab-ipadic-neologd'


class DialogueAgent:
    def __init__(self):
        self.tagger = MeCab.Tagger('-d {}'.format(MECAB_DIC_DIR))

    def _tokenize(self, text):
        node = self.tagger.parseToNode(text)

        tokens = []
        while node:
            if node.surface != '':
                tokens.append(node.surface)

            node = node.next

        return tokens

    def train(self, texts, labels):
        vectorizer = CountVectorizer(tokenizer=self._tokenize)
        bow = vectorizer.fit_transform(texts)  # <2>

        classifier = SVC()
        classifier.fit(bow, labels)

        # <3>
        self.vectorizer = vectorizer
        self.classifier = classifier

    def predict(self, texts):
        bow = self.vectorizer.transform(texts)
        return self.classifier.predict(bow)


if __name__ == '__main__':
    BASE_DIR = normpath(dirname(__file__))

    training_data = pd.read_csv(join(BASE_DIR, './training_data.csv'))  # <4>

    dialogue_agent = DialogueAgent()
    dialogue_agent.train(training_data['text'], training_data['label'])

    with open(join(BASE_DIR, './replies.csv')) as f:  # <5>
        replies = f.read().split('\n')

    input_text = '名前を教えてよ'
    predictions = dialogue_agent.predict([input_text])  # <6>
    predicted_class_id = predictions[0]  # <7>

    print(replies[predicted_class_id])

    while True:
        input_text = input()
        predictions = dialogue_agent.predict([input_text])
        predicted_class_id = predictions[0]

        print(replies[predicted_class_id])
