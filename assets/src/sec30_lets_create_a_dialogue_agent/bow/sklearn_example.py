from sklearn.feature_extraction.text import CountVectorizer

from tokenizer import tokenize  # <1>

texts = [
    '私は私のことが好きなあなたが好きです',
    '私はラーメンが好きです。',
    '富士山は日本一高い山です',
]

# Bag of Words計算
vectorizer = CountVectorizer(tokenizer=tokenize)  # <2>
vectorizer.fit(texts)  # <3>
bow = vectorizer.transform(texts)  # <4>
