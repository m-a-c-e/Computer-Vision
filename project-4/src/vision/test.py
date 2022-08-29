import numpy as np
import torch
from torchvision.models import resnet18
import torch.nn as nn


a = np.array([[1, 2, 3], [1, 2, 3]])
b = np.array([[1], [2]])
print(a/b)

print(np.mean(a, axis=0))