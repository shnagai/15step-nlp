import gensim.downloader as api

model = api.load('glove-wiki-gigaword-50')  # <1>

tokyo = model['tokyo']  # <2>
japan = model['japan']
france = model['france']

v = tokyo - japan + france  # <3>

print('tokyo - japan + france = ',
      model.wv.similar_by_vector(v, topn=1)[0])  # <4>
