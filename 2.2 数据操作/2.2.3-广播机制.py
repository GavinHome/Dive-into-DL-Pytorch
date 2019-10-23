#%% [markdown]
import torch
x = torch.arange(1,3).view(1,2)
print(x)
y = torch.arange(1,4).view(3,1)
print(y)
print(x+y)
#%% [markdown]
# 由于 x 和 y 分别是1⾏2列和3⾏1列的矩阵，如果要计算 x + y ，那么 x 中第⼀⾏的2个元素被⼴播
# （复制）到了第⼆⾏和第三⾏，⽽ y 中第⼀列的3个元素被⼴播（复制）到了第⼆列。如此，就可以对2
# 个3⾏2列的矩阵按元素相加。
#%% [markdown]