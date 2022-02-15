import numpy as np
import torch

a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
c = np.array([[1, 2, 3], [4, 75, 6]], dtype=np.float32)
d = torch.from_numpy(a)
d = d.detach().cpu().numpy()
print(d)

b = np.where(a == c, 0, a)
print(b)

