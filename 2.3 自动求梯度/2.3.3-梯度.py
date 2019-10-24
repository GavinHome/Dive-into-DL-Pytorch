
import torch

#%% [markdown]
# 创建Tensor设置requires_grad=True
x = torch.ones(2,2, requires_grad=True)
y = x + 2
#%% [markdown]
# $$z=3(x+2)^2 $$
z = y * y * 3
#%% [markdown]
# $$out= \frac 1n \sum_{i=1}^n 3(x_i+2)^2, n = 4$$
out = z.mean()
out.backward()
print(x.grad)
#%% [markdown]
# out关于x的梯度
# $$ \frac {d(o)}{d_{x}}= \{\frac {1}{4} \sum_{i=1}^4 3(x_i+2)^2\}^{'}=\frac {1}{4}*3*2*(x_{i}+2)$$
# $$ \frac {d(o)}{d_{x}}= \frac {3x_{i}+6}{2}$$
# $$ \frac {d(o)}{d_{x_{i}}}|_{x_{i}=1}= \frac {3*1+6}{2} = \frac {9}{2}=4.5$$
#%% [markdown]
out2 = x.sum()
out2.backward()
print(x.grad)

out3 = x.sum()
x.grad.data.zero_()
out3.backward()
print(x.grad)
#%% [markdown]
x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
y = 2 * x
z = y.view(2, 2)
print(y)
print(z)
#%% [markdown]
v = torch.tensor([[1.0, 0.1], [0.01, 0.001]], dtype=torch.float)
z.backward(v)
print(x.grad)
#%% [markdown]
# 中断梯度追踪
x = torch.tensor(1.0, requires_grad=True)
y1 = x ** 2
with torch.no_grad():
    y2 = x ** 3
y3 = y1 + y2
print(x.requires_grad)
print(y1, y1.requires_grad) # True
print(y2, y2.requires_grad) # False
print(y3, y3.requires_grad) # True
y3.backward()
print(x.grad)
#%% [markdown]
# 如果我们想要修改 tensor 的数值，但是⼜不希望被 autograd 记录（即不会影响反向传播），
# 那么我么可以对 tensor.data 进⾏操作。
x = torch.ones(1,requires_grad=True)
print(x.data) # 还是⼀个tensor
print(x.data.requires_grad) # 但是已经是独⽴于计算图之外
y = 2 * x
x.data *= 100 # 只改变了值，不会记录在计算图，所以不会影响梯度传播
y.backward()
print(x) # 更改data的值也会影响tensor的值
print(x.grad)


#%% [markdown]
