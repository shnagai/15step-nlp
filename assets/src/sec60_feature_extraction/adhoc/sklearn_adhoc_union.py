import re

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline

rx_periods = re.compile(r'[.。．]+')  # <1>


class TextStats(BaseEstimator, TransformerMixin):  # <2>
    def fit(self, x, y=None):  # <3>
        return self

    def transform(self, texts):  # <4>
        return [
            {
                'length': len(text),
                'num_sentences': len([sent for sent in rx_periods.split(text)
                                      if len(sent) > 0])
            }
            for text in texts
        ]


combined = FeatureUnion([  # <5>
    ('stats', Pipeline([
        ('stats', TextStats()),
        ('vect', DictVectorizer()),  # <6>
    ])),
    ('char_bigram', CountVectorizer(analyzer='char', ngram_range=(2, 2))),
])

texts = [
    'こんにちは。こんばんは。',
    '焼肉が食べたい'
]

combined.fit(texts)
feat = combined.transform(texts)
