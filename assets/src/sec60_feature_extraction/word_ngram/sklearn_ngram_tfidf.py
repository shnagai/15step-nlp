from sklearn.feature_extraction.text import TfidfVectorizer

from tokenizer import tokenize

texts = [
    '東京から大阪に行く',
    '大阪から東京に行く',
]

# bi-gram
vectorizer = TfidfVectorizer(tokenizer=tokenize, ngram_range=(2, 2))
vectorizer.fit(texts)
tfidf = vectorizer.transform(texts)
