import torch
#%% [markdown]
# ⽅法 to() 可以将 Tensor 在CPU和GPU（需要硬件⽀持）之间相互移动。
#%% [markdown]
# 以下代码只有在PyTorch GPU版本上才会执⾏
if torch.cuda.is_available():
    device = torch.device("cuda") # GPU
    y = torch.ones_like(x, device=device) # 直接创建⼀个在GPU上的Tensor
    x = x.to(device) # 等价于 .to("cuda")
    z = x + y
    print(z)
    print(z.to("cpu", torch.double)) # to()还可以同时更改数据类型

#%% [markdown]