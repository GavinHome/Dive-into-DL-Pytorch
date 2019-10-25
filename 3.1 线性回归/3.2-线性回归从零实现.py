#%% [markdown]
import torch
from matplotlib import pyplot as plt
import numpy as np
import sys
sys.path.append("..")
from d2lzh_pytorch import *

#%% [markdown]
# 1.生成数据集
num_inputs = 2
num_examples = 1000
true_w = [2,-3.4]
true_b = 4.2
features = torch.from_numpy(np.random.normal(0,1,(num_examples,num_inputs)))
labels = true_w[0] * features[:,0] + true_w[1] * features[:,1] + true_b
labels += torch.from_numpy(np.random.normal(0,0.01,size=labels.size()))
print(features[0], labels[0])
#%% [markdown]
# ⽣成第⼆个特征 features[:, 1] 和标签 labels 的散点图，可以更直观地观察两者间的线性关系
set_figsize()
plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
#%% [markdown]
# 2.读取数据
batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, y)
    break

#%% [markdown]
# 3.初始化模型参数
w = torch.tensor(np.random.normal(0,0.01,(num_inputs,1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

#%% [markdown]
# 4.定义模型
# 线性回归的⽮量计算表达式: $$ y = Xw + b$$, 使⽤ mm 函数做矩阵乘法, 此函数已保存在d2lzh_pytorch包中⽅便以后使⽤
def linreg(X, w, b):
    return torch.mm(X, w) + b

#%% [markdown]
# 5.定义损失函数
# 平⽅损失来定义线性回归的损失函数, 本函数已保存在d2lzh_pytorch包中⽅便以后使⽤
def squared_loss(y_hat, y):
    # 注意这⾥返回的是向量, 另外, pytorch⾥的MSELoss并没有除以 2
    return (y_hat - y.view(y_hat.size())) ** 2 / 2

#%% [markdown]
# 6.定义优化算法
# ⼩批量随机梯度下降算法: 通过不断迭代模型参数来优化损失函数, 本函数已保存在d2lzh_pytorch包中⽅便以后使⽤
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size # 注意这里更改param时用的param.data

#%% [markdown]
# 7.训练模型
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss
for epoch in range(num_epochs): # 训练模型⼀共需要num_epochs个迭代周期
    # 在每⼀个迭代周期中，会使⽤训练数据集中所有样本⼀次（假设样本数能够被批量⼤⼩整除） 
    # X和y分别是⼩批量样本的特征和标签
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X.double(), w.double(), b.double()), y).sum() # l是有关⼩批量X和y的损失
        l.backward() # ⼩批量的损失对模型参数求梯度
        sgd([w, b], lr, batch_size) # 使⽤⼩批量随机梯度下降迭代模型参数

        # 不要忘了梯度清零
        w.grad.data.zero_()
        b.grad.data.zero_()
    train_l = loss(net(features.double(), w.double(), b.double()), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))

print(true_w, '\n', w)
print(true_b, '\n', b)

#%% [markdown]
# 小结