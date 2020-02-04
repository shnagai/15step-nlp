from os.path import dirname, join, normpath

import MeCab
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


class DialogueAgent:
    def __init__(self):
        self.tagger = MeCab.Tagger()

    def _tokenize(self, text):
        node = self.tagger.parseToNode(text)

        tokens = []
        while node:
            if node.surface != '':
                tokens.append(node.surface)

            node = node.next

        return tokens

    def train(self, texts, labels):
        pipeline = Pipeline([
            ('vectorizer', CountVectorizer(tokenizer=self._tokenize)),
            ('svd', TruncatedSVD(n_components=100)),
            ('classifier', SVC()),
        ])

        pipeline.fit(texts, labels)

        self.pipeline = pipeline

    def predict(self, texts):
        return self.pipeline.predict(texts)


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
