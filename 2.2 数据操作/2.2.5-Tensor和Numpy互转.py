import torch
#%% [markdown]
# 1. Tensor to Numpy
a = torch.ones(5)
b = a.numpy()
print(a,b)
a += 1
print(a,b)
b += 1
print(a,b)
#%% [markdown]
# 2. Numpy to Tensor
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
print(a, b)
a += 1
print(a, b)
b += 1
print(a, b)
#%% [markdown]
# numpy() 和 from_numpy() 这两个函数所产⽣的的 Tensor 和NumPy中的数组共享相同的内存（所以他们之间的转换很快），改变其中⼀个时另⼀个也会改变！！！
#%% [markdown]
# torch.tensor() 将NumPy数组转换为tensor, 但是总是会进⾏数据拷⻉，返回的 Tensor 和原来的数据不再共享内存
c = torch.tensor(a)
a += 1
print(a, c)
#%% [markdown]