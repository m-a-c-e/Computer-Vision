import torch
from torch import nn
import numpy as np


a = np.array([[1, 2, 3], [4, 5, 7], [8, 9, 19]], dtype=np.float32)

print(a)
a = np.reshape(a, (1, 3, 3))

a_tensor = torch.from_numpy(a)

conv = nn.MaxPool2d((3, 3),  stride=(1, 1), padding=[1, 1])
output = conv(a_tensor)

output = output.detach().cpu().numpy()
output = np.resize(output, (output.shape[1], output.shape[2]))


print(output)
print(np.array([1, 3]))
