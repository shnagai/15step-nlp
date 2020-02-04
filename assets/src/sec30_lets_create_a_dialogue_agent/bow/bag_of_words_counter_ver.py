from collections import Counter

from tokenizer import tokenize


def calc_bow(tokenized_texts):
    counts = [Counter(tokenized_text)
              for tokenized_text in tokenized_texts]  # <1>
    sum_counts = sum(counts, Counter())  # <2>
    vocabulary = sum_counts.keys()

    bow = [[count[word] for word in vocabulary]
           for count in counts]  # <3>

    return bow


# 入力文のlist
texts = [
    '私は私のことが好きなあなたが好きです',
    '私はラーメンが好きです',
    '富士山は日本一高い山です',
]

tokenized_texts = [tokenize(text) for text in texts]
bow = calc_bow(tokenized_texts)
