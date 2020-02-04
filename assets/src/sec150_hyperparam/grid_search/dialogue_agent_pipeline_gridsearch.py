"""
isort:skip_file
"""
from os.path import normpath, dirname, join
import MeCab
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import pandas as pd
import unicodedata
import neologdn


tagger = MeCab.Tagger()


def tokenize(text):
    text = unicodedata.normalize('NFKC', text)  # <1>
    text = neologdn.normalize(text)  # <2>
    text = text.lower()  # <3>

    node = tagger.parseToNode(text)
    result = []
    while node:
        features = node.feature.split(',')

        if features[0] != 'BOS/EOS':
            if features[0] not in ['助詞', '助動詞']:  # <4>
                token = features[6] \
                        if features[6] != '*' \
                        else node.surface  # <5>
                result.append(token)

        node = node.next

    return result


class DialogueAgent:
    def __init__(self):
        self.tagger = MeCab.Tagger()

    def _tokenize(self, text):
        text = unicodedata.normalize('NFKC', text)  # <1>
        text = neologdn.normalize(text)  # <2>
        text = text.lower()  # <3>

        node = self.tagger.parseToNode(text)
        result = []
        while node:
            features = node.feature.split(',')

            if features[0] != 'BOS/EOS':
                if features[0] not in ['助詞', '助動詞']:  # <4>
                    token = features[6] \
                            if features[6] != '*' \
                            else node.surface  # <5>
                    result.append(token)

            node = node.next

        return result

    def train(self, texts, labels):
        pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(tokenizer=tokenize,
                                           ngram_range=(1, 2))),
            ('classifier', RandomForestClassifier()),
        ])

        parameters = {
            'vectorizer__ngram_range':
                [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)],
            'classifier__n_estimators':
                [10, 20, 30, 40, 50, 100, 200, 300, 400, 500],
            'classifier__max_features':
                ('sqrt', 'log2', None),
        }
        clf = GridSearchCV(pipeline, parameters)

        clf.fit(texts, labels)

        self.clf = clf

    def predict(self, texts):
        return self.clf.predict(texts)


if __name__ == '__main__':
    BASE_DIR = normpath(dirname(__file__))

    training_data = pd.read_csv(join(BASE_DIR, './training_data.csv'))

    dialogue_agent = DialogueAgent()
    dialogue_agent.train(training_data['text'], training_data['label'])

    with open(join(BASE_DIR, './replies.csv')) as f:
        replies = f.read().split('\n')

    input_text = '名前は？'
    predictions = dialogue_agent.predict([input_text])
    predicted_class_id = predictions[0]

    print(replies[predicted_class_id])

    while True:
        input_text = input()
        predictions = dialogue_agent.predict([input_text])
        predicted_class_id = predictions[0]

        print(replies[predicted_class_id])
