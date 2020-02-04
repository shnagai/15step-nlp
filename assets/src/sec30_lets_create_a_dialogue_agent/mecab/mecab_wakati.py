import MeCab

tagger = MeCab.Tagger('-Owakati')

print(tagger.parse('私は私のことが好きなあなたが好きです'))

assert tagger.parse('私は私のことが好きなあなたが好きです') == '私 は 私 の こと が 好き な あなた が 好き です \n'
