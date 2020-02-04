from gensim.models import Word2Vec

from cnn_sample import tokens_to_sequence

we_model = Word2Vec.load(
    './latest-ja-word2vec-gensim-model/word2vec.gensim.model')

print(tokens_to_sequence(we_model, ['あなた', 'の', '名前', 'を', '教えて', 'よ']))
