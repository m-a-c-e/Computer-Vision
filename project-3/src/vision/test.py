import numpy as np


a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
b = np.array([[1, 1, 1], [0, 0, 2]])

print(a[:, :-1])

c = np.dot(b, a)
d = np.matmul(b, a)
print(a.shape)
print(b.shape)
print(c)
print(d)
print(c.shape)