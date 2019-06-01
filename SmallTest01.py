# import modules
import numpy as np
import matplotlib.pyplot as plt

from frame_2D_alg.utils import imread, draw

# read image
image = imread('./images/raccoon_eye.jpg').astype(int)
Y, X = image.shape

# define kernels
k1 = np.array([[0, 0, 0],
               [-1, 0, 1],
               [0, 0, 0]]).reshape((1, 3, 3))

k2 = np.array([[0, -1, 0],
               [0, 0, 0],
               [0, 1, 0]]).reshape((1, 3, 3))

k = np.concatenate((k1, k2), axis=0)

derivatives = np.empty((2, Y, X), dtype=int)

for y in range(0, Y - 2):
    for x in range(0, X - 2):
        derivatives[:, y, x] = (image[y:y+3, x:x + 3] * k).sum(axis=(1, 2))

derivatives = derivatives // 2

derivatives += 128

draw('./debug/derts1', derivatives[0, :, :])
draw('./debug/derts2', derivatives[1, :, :])

# plt.imshow(derivatives[1, :, :])
# plt.show()