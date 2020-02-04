from keras.preprocessing.sequence import pad_sequences  # noqa: F401

sequences = [  # 長さの異なるSequenceのList
    [1, 2, 3],
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
]
