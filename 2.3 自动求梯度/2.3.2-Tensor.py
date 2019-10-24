
import torch

#%% [markdown]
# 创建Tensor设置requires_grad=True
x = torch.ones(2,2, requires_grad=True)
print(x)
print(x.grad_fn)

#%% [markdown]
# 运算操作
y = x + 2
print(y)
print(y.grad_fn)

#%% [markdown]
# x这种直接创建的称为叶⼦节点，叶⼦节点对应的 grad_fn 是 None;
# y是通过⼀个加法操作创建的，所以它有⼀个为<AddBackward> 的 grad_fn
print(x.is_leaf, y.is_leaf)

#%% [markdown]
# $$z=3(x_i+2)^2 $$
z = y * y * 3
print(z)
#%% [markdown]
# $$out= \frac 1n \sum_{i=1}^n 3(x_i+2)^2, n = 4$$
out = z.mean()
print(z, out)

#%% [markdown]
# 通过 .requires_grad_() 来⽤in-place的⽅式改变 requires_grad 属性
a = torch.randn(2,2)
a = ((a*3) / (a-1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a*a).sum()
print(b.grad_fn)

#%% [markdown]
