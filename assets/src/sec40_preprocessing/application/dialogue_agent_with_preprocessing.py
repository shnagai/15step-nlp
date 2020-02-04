"""
isort:skip_file
"""
from os.path import normpath, dirname, join
import MeCab
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
import pandas as pd
import unicodedata
import neologdn


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

    input_text = '名前は？'
    predictions = dialogue_agent.predict([input_text])
    predicted_class_id = predictions[0]

    print(replies[predicted_class_id])

    while True:
        input_text = input()
        predictions = dialogue_agent.predict([input_text])
        predicted_class_id = predictions[0]

        print(replies[predicted_class_id])
