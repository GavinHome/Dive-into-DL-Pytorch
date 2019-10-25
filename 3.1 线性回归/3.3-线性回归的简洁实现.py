#%% [markdown]
import torch
from torch import nn
import numpy as np
import torch.utils.data as Data
torch.manual_seed(1)
print(torch.__version__)
torch.set_default_tensor_type('torch.FloatTensor')

#%% [markdown]
# 1.生成数据集
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

#%% [markdown]
# 2.读取数据
batch_size = 10
# 将训练数据的特征和标签组合
dataset = Data.TensorDataset(features, labels)
# 把 dataset 放入 DataLoader
data_iter = Data.DataLoader(
    dataset=dataset,            # torch TensorDataset format
    batch_size=batch_size,      # mini batch size
    shuffle=True                # 要不要打乱数据 (打乱比较好)
)

for X, y in data_iter:
    print(X, y)
    break

#%% [markdown]
# 3.定义模型
class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)

    def forward(self, x):
        y = self.linear(x)
        return y
    
net = LinearNet(num_inputs)
print(net) # 使用print可以打印出网络的结构

for param in net.parameters():
    print(param)

#%% [markdown]
# 写法一
net = nn.Sequential(
    nn.Linear(num_inputs, 1)
    # 此处还可以传入其他层
    )

#%% [markdown]
# 写法二
net = nn.Sequential()
net.add_module('linear', nn.Linear(num_inputs, 1))

#%% [markdown]
# 写法三
from collections import OrderedDict
net = nn.Sequential(OrderedDict([
          ('linear', nn.Linear(num_inputs, 1))
          # ......
        ]))


#%% [markdown]
print(net)
print(net[0])

#%% [markdown]
# 4.初始化模型参数
from torch.nn import init
init.normal_(net[0].weight, mean=0.0, std=0.01)
init.constant_(net[0].bias, val=0.0)  # 也可以直接修改bias的data: net[0].bias.data.fill_(0)

for param in net.parameters():
    print(param)

#%% [markdown]
# 5.定义损失函数
loss = nn.MSELoss()

#%% [markdown]
# 6.定义优化算法
optimizer = torch.optim.SGD(net.parameters(), lr=0.03)
print(optimizer)

# 为不同子网络设置不同的学习率
# optimizer =optim.SGD([
#                 # 如果对某个参数不指定学习率，就使用最外层的默认学习率
#                 {'params': net.subnet1.parameters()}, # lr=0.03
#                 {'params': net.subnet2.parameters(), 'lr': 0.01}
#             ], lr=0.03)


# 调整学习率
# for param_group in optimizer.param_groups:
#     param_group['lr'] *= 0.1 # 学习率为之前的0.1倍

#%% [markdown]
# 7.训练模型
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad() # 梯度清零，等价于net.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))

#%% [markdown]
dense = net[0]
print(true_w, dense.weight)
print(true_b, dense.bias)

# %%
