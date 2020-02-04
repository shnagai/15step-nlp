from gensim.models import Word2Vec

model = Word2Vec.load('./ja/ja.bin')

tokyo = model['東京']
france = model['フランス']
japan = model['日本']

print(model.wv.similar_by_vector(tokyo - japan + france))
