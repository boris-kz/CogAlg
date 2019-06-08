from itertools import chain, starmap

a = range(4)

b = [(a, i) for i in range(3)]

print([y for x, y in chain(*starmap(enumerate, b)) if x % 2])

# for x in starmap(enumerate, b):
#     print([*x])