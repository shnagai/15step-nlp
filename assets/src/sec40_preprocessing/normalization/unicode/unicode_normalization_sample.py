import unicodedata

normalized = unicodedata.normalize('NFKC', '㈱リックテレコム')

assert normalized == '(株)リックテレコム'
print(normalized)
