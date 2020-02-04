from sklearn.feature_extraction.text import TfidfVectorizer  # <1>

from tokenizer import tokenize

texts = [
    '私は私のことが好きなあなたが好きです',
    '私はラーメンが好きです。',
    '富士山は日本一高い山です',
]

# TF-IDF計算
vectorizer = TfidfVectorizer(tokenizer=tokenize)
vectorizer.fit(texts)
tfidf = vectorizer.transform(texts)
