import numpy as np

def m_mul(a, b):

    c = [[0 for _ in range(len(a))] for _ in range(len(b))]

    for i in range(len(b)):
        for j in range(len(a[0])):
            c[i][j] = sum([b[i][k] * a[k][j] for k in range(len(a))])

    return c


a = [[1, 2],
     [3, 4]]

b = [[5, 6],
     [7, 8]]

c = [[9, 10],
     [11, 12]]

M = np.array([[(1 + 1j) / 2 ** 0.5,     0                  ],
              [0,                       (1 - 1j) / 2 ** 0.5]])

S = np.array([[1j, -1j],
              [1, 1]])

S_inv = np.linalg.inv(S)

print(S_inv)

print(np.dot(S_inv, np.dot(M, S)))