import MeCab

tagger = MeCab.Tagger()


def tokenize(text, stop_words):  # <1>
    node = tagger.parseToNode(text)
    result = []
    while node:
        features = node.feature.split(',')

        if features[0] != 'BOS/EOS':
            token = features[6] if features[6] != '*' else node.surface
            if token not in stop_words:  # <2>
                result.append(token)

        node = node.next

    return result


stop_words = ['て', 'に', 'を', 'は', 'です', 'ます']  # <3>

print(tokenize('本を読んだ', stop_words))
print(tokenize('本を読みました', stop_words))
