import MeCab

tagger = MeCab.Tagger()


def tokenize(text):
    node = tagger.parseToNode(text)

    tokens = []
    while node:
        if node.surface != '':
            tokens.append(node.surface)

        node = node.next

    return tokens
