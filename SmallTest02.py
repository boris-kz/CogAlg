import numpy as np

np.random.seed(0)

a = np.arange(10).reshape(5, 2)

print(a)

print(np.swapaxes(a, 0, 1))