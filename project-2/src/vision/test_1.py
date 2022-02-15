from numpy import dtype
import torch
from torch import nn
import numpy as np

a = np.array([1, 2, 1], dtype=np.float32)
c = np.where(a == 1)[0][1]
print(c)