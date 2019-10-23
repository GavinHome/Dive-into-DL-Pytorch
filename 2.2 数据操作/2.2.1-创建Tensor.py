#%% [markdown]
# 1. 导入PyTorch
import torch
#%% [markdown]
# 2. 创建5*3未初始化的Tensor
x = torch.empty(5,3)
#%% [markdown]
# 3. 输出Tensor
print(x)
#%% [markdown]
# 4. 创建5*3初始化的Tensor
x = torch.rand(5,3)
#%% [markdown]
# 5. 输出Tensor
print(x)
#%% [markdown]
# 6. 创建5*3的long型全是0的Tensor
x = torch.zeros(5,3, dtype=torch.long)
#%% [markdown]
# 7. 输出Tensor
print(x)
#%% [markdown]
# 8. 创建5*3的long型全是0的Tensor
x = torch.tensor([5.5,3])
#%% [markdown]
# 9. 输出Tensor
print(x)
#%% [markdown]
# 10. 创建自定义数据类型的Tensor
x = x.new_ones(5,3, dtype=torch.float64) #返回的tensor默认具有相同的dtype和device
#%% [markdown]
# 11. 输出Tensor
print(x)
#%% [markdown]
# 12. 试用已有Tensor创建，指定新的数据类型
x = torch.randn_like(x, dtype=torch.float) #指定新的数据类型
#%% [markdown]
# 13. 输出Tensor
print(x)
#%% [markdown]
# 14. shape或size()获取Tensor的形状
print(x.shape)
print(x.size())
#%% [markdown]
# 15. 输出Tensor
print(x)
#%% [markdown]
# 16. 其他创建Tensor的API
#%% [markdown]
# 函数 | 功能 
# :-: | :-: | :-: | :-: | :-:
# Tensor(*sizes) | 基础构造函数 | 
# tensor(data) | 类似np.array的构造函数 | 
# ones(*sizes) | 全1Tensor | 
# zeros(*sizes) | 全0Tensor | 
# eye(*sizes) | 对角线为1, 其他为0 |
# arange(s.e.step) | 从s到e,步长为step |
# linspace(s.e.steps) | 从s到e,均匀切分成steps份 |
# rand/randn(*sizes) | 均匀/标准分布 | 
# normal(mean,std)/uniform(from,to) | 正态分布/均匀分布 | 
# randperm(m) | 随机排列 | 
#%% [markdown]
