#%% [markdown]
# 导入torch，准备数据
import torch
x = torch.rand(5,3)
y = torch.rand(5,3)
#%% [markdown]
# 1. 算术操作-加法1
print(x+y)
#%% [markdown]
# 加法2
print(torch.add(x,y))
result = torch.empty(5,3)
torch.add(x,y,out=result)
print(result)
#%% [markdown]
# 加法3
y.add_(x)
print(y)
#%% [markdown]
# 2. 索引
print(x)
y1 = x[0,:]
print(y)
y1 += 1
print(y1)
print(x[0,:])
#%% [markdown]
# 高级索引API
#%% [markdown]
# 函数 | 功能 
# :-: | :-: | :-: | :-: | :-:
# index_select(input,dim,index) | 在指定维度dim上选取，比如选取某些行，某些列 | 
# masked_select(input,mask) | 例子如上，a(a>0)，使用ByteTensor进行选取 | 
# non_zero(input) | 非0元素的下标 | 
# gather(input,dim,index) | 根据index，在dim维度上选取数据，输出的size与index一样 | 
#%% [markdown]
# 3. 改变形状
y2 = x.view(15)
z= x.view(-1,5)
print(x.size(), y2.size(), z.size())
#%% [markdown]
# view只改变张量观察角度
x += 1
print(x)
print(y2)
print(z)
#%% [markdown]
# 创建x的副本
x_cp= x.clone().view(15)
x -= 1
print(x)
print(x_cp)
#%% [markdown]
# item()
x1 = torch.rand(1)
y3 = x1.item()
print(x1)
print(y3)
#%% [markdown]
# 4. 线性代数
#%% [markdown]
# 函数 | 功能 
# :-: | :-: | :-: | :-: | :-:
# trace | 对角线元素之和（矩阵的迹） | 
# diag | 对角线元素 | 
# triu/tril | 矩阵的上三角/下三角,可指定偏移量 | 
# mm/bmm | 矩阵乘法，batch的矩阵乘法 | 
# addmm/addbmm/addmv/addr/badbmm | 矩阵运算 | 
# t | 转置 | 
# dot/cross | 内积/外积 | 
# inverse | 求逆矩阵 | 
# svd | 奇异值分解 | 

#%% [markdown]

