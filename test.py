import itertools
a = [["the quick brown fox jumped over the lazy dog", "bbig bob joe"], ["c f g d "]]

print(list(itertools.chain.from_iterable(a)))