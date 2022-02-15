import numpy as np

a = np.array([[1, 0, 0], [4, 4, 0]])

b = np.where(a != 0)

b_x = b[0]
b_y = b[1]
print(b_x, " and ", b_y)

c = a[b_x, b_y]
print(c)