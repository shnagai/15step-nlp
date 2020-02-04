import MeCab

tagger = MeCab.Tagger()


def tokenize(text):
    node = tagger.parseToNode(text)
    result = []
    while node:
        features = node.feature.split(',')  # <1>

        if features[0] != 'BOS/EOS':  # <2>
            token = features[6] if features[6] != '*' else node.surface  # <1>
            result.append(token)

        node = node.next

    return result


print(tokenize('本を読んだ'))
print(tokenize('本を読みました'))
