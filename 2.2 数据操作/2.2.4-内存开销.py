import torch

#%% [markdown]
# 索引、 view 是不会开辟新内存的，⽽像 y = x + y 这样的运算是会新开内存的，然后将 y 指向新内存。
# Python⾃带 id 函数：如果两个实例的ID⼀致，那么它们所对应的内存地址相同；反之则不同。
x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
y = y + x
print(id(y) == id_before) # False

#%% [markdown]
# 如果想指定结果到原来的 y 的内存，我们可以使⽤前⾯介绍的索引来进⾏替换操作。在下⾯的例⼦中，
# 我们把 x + y 的结果通过 [:] 写进 y 对应的内存中。
x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
y[:] = y + x
print(id(y) == id_before) # True

#%% [markdown]
# 我们还可以使⽤运算符全名函数中的 out 参数或者⾃加运算符 += (也即 add_() )达到上述效果，例如torch.add(x, y, out=y) 和 y += x ( y.add_(x) )。
x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
torch.add(x, y, out=y) # y += x, y
print(id(y) == id_before) # True


#%%
