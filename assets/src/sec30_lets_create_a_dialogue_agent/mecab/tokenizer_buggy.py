import MeCab

tagger = MeCab.Tagger('-Owakati')


def tokenize(text):
    return tagger.parse(text).strip().split(' ')
