import scipy  # <1>
from sklearn.feature_extraction.text import CountVectorizer

from tokenizer import tokenize

texts = [
    '私は私のことが好きなあなたが好きです',
    '私はラーメンが好きです。',
    '富士山は日本一高い山です',
]

# <2>
word_bow_vectorizer = CountVectorizer(tokenizer=tokenize)
char_bigram_vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 2))

# <3>
word_bow_vectorizer.fit(texts)
char_bigram_vectorizer.fit(texts)

# <4>
word_bow = word_bow_vectorizer.transform(texts)
char_bigram = char_bigram_vectorizer.transform(texts)

feat = scipy.sparse.hstack((word_bow, char_bigram))  # <5>
