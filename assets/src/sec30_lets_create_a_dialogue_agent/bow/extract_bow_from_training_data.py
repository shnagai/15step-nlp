from os.path import dirname, join, normpath

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from tokenizer import tokenize  # <1>

# データ読み込み
BASE_DIR = normpath(dirname(__file__))
csv_path = join(BASE_DIR, './training_data.csv')  # <2>
training_data = pd.read_csv(csv_path)  # <3>
training_texts = training_data['text']

# Bag of Words計算  <4>
vectorizer = CountVectorizer(tokenizer=tokenize)
vectorizer.fit(training_texts)
bow = vectorizer.transform(training_texts)
