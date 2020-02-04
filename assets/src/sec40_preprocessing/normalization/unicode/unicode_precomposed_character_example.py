from sklearn.feature_extraction.text import CountVectorizer

from tokenizer import tokenize

texts = [
    'ディスプレイを買った',
    'ディスプレイを買った',
]

vectorizer = CountVectorizer(tokenizer=tokenize)
vectorizer.fit(texts)

print('bow:')
print(vectorizer.transform(texts).toarray())

print('vocabulary:')
print(vectorizer.vocabulary_)
