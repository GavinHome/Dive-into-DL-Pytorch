#%% [markdown]
import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

#%% [markdown]
# 1.读取数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=None, root="Datasets")
#%% [markdown]
# 2.定义和初始化模型
num_inputs = 784
num_outputs = 10

class LinearNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)
    
    def forward(self, x):
        y = self.linear(x.view(x.shape[0], -1))
        return y
net = LinearNet(num_inputs, num_outputs)
init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)

#%% [markdown]
# 3.定义损失函数
loss = nn.CrossEntropyLoss()

#%% [markdown]
# 4.定义优化算法
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

#%% [markdown]
# 5.训练模型
num_epochs = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)

#%% [markdown]
# 6.预测
X, y = iter(test_iter).next()
true_labels = d2l.get_fashion_mnist_labels(y.numpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels,pred_labels)]
d2l.show_fashion_mnist(X[0:9], titles[0:9])

#%% [markdown]

