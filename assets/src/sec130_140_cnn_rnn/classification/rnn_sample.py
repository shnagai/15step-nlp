import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from keras.layers import LSTM, Dense, Embedding
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from sklearn.metrics import accuracy_score

from tokenizer import tokenize


def tokens_to_sequence(we_model, tokens):
    sequence = []
    for token in tokens:
        try:
            sequence.append(we_model.wv.vocab[token].index + 1)
        except KeyError:
            pass
    return sequence


def get_keras_embedding(keyed_vectors, *args, **kwargs):
    weights = keyed_vectors.vectors
    word_num = weights.shape[0]
    embedding_dim = weights.shape[1]
    zero_word_vector = np.zeros((1, weights.shape[1]))
    weights_with_zero = np.vstack((zero_word_vector, weights))
    return Embedding(input_dim=word_num + 1,
                     output_dim=embedding_dim,
                     weights=[weights_with_zero],
                     *args, **kwargs)


if __name__ == '__main__':
    we_model = Word2Vec.load(
        './latest-ja-word2vec-gensim-model/word2vec.gensim.model')  # <1>

    # 学習データ読み込み
    training_data = pd.read_csv('training_data.csv')

    # 学習データのテキストを分かち書きし、トークン化→インデックス化
    training_texts = training_data['text']
    tokenized_training_texts = [tokenize(text) for text in training_texts]
    training_sequences = [tokens_to_sequence(we_model, tokens)
                          for tokens in tokenized_training_texts]

    # training_sequencesの全要素の長さをMAX_SEQUENCE_LENGTHに揃える
    MAX_SEQUENCE_LENGTH = 20
    x_train = pad_sequences(training_sequences, maxlen=MAX_SEQUENCE_LENGTH)

    # 学習データとしてクラスIDのリストを準備
    y_train = np.asarray(training_data['label'])
    n_classes = max(y_train) + 1

    # モデル構築 <5>
    model = Sequential()
    model.add(get_keras_embedding(we_model.wv,
                                  input_shape=(MAX_SEQUENCE_LENGTH, ),
                                  mask_zero=True,
                                  trainable=False))
    model.add(LSTM(units=256))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=n_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    # 学習
    model.fit(x_train, to_categorical(y_train), epochs=50)

    # ===== 評価 ===== <8>

    # 学習データと同様にテストデータを準備
    test_data = pd.read_csv('test_data.csv')

    test_texts = test_data['text']
    tokenized_test_texts = [tokenize(text) for text in test_texts]
    test_sequences = [tokens_to_sequence(we_model, tokens)
                      for tokens in tokenized_test_texts]
    x_test = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

    y_test = np.asarray(test_data['label'])

    # 予測
    y_pred = np.argmax(model.predict(x_test), axis=1)

    # 評価
    print(accuracy_score(y_test, y_pred))
