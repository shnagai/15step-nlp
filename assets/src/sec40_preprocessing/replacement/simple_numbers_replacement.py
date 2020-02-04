import re


def tokenize_numbers(text):
    return re.sub(r'\d+', ' SOMENUMBER ', text)


print(tokenize_numbers('卵を1個買ったよ！'))
print(tokenize_numbers('卵を2個買ったよ！'))
print(tokenize_numbers('卵を10個買ったよ！'))
