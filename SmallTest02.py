import numpy as np

np.random.seed(0)

a = np.arange(9).reshape(3, 3)

b = (a % 2).nonzero()

a[b] = -1

print(a)