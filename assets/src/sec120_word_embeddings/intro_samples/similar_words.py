import gensim.downloader as api

model = api.load('glove-wiki-gigaword-50')  # <1>

tokyo = model['tokyo']  # <2>
print(model.wv.similar_by_vector(tokyo))
