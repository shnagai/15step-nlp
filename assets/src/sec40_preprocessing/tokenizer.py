import unicodedata

import MeCab
import neologdn

tagger = MeCab.Tagger()


def tokenize(text):
    text = unicodedata.normalize('NFKC', text)  # <1>
    text = neologdn.normalize(text)  # <2>

    node = tagger.parseToNode(text)
    result = []
    while node:
        features = node.feature.split(',')

        if features[0] != 'BOS/EOS':
            if features[0] not in ['助詞', '助動詞']:  # <4>
                token = features[6] \
                        if features[6] != '*' \
                        else node.surface  # <5>
                result.append(token)

        node = node.next

    return result
