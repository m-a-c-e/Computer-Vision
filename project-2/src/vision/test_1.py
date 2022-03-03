from traceback import print_tb
import numpy as np

data = np.array([[1, 1, 4, 4], [3, 43, 8, 9]], dtype=np.float32)
min_arr = np.min(data, axis=1)

#
for i in range(0, data.shape[0]):
    idx = np.where(data[i] == min_arr[i])[0][0]
    data[i][idx] = float('inf')

min_arr_2 = np.min(data, axis=1)

print(min_arr)
print(min_arr_2)

    
