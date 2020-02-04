from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import FeatureUnion  # <1>

from tokenizer import tokenize

texts = [
    '私は私のことが好きなあなたが好きです',
    '私はラーメンが好きです。',
    '富士山は日本一高い山です',
]

word_bow_vectorizer = CountVectorizer(tokenizer=tokenize)
char_bigram_vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 2))

# <2>
estimators = [
    ('bow', word_bow_vectorizer),
    ('char_bigram', char_bigram_vectorizer),
]
combined = FeatureUnion(estimators)
combined.fit(texts)
feat = combined.transform(texts)
