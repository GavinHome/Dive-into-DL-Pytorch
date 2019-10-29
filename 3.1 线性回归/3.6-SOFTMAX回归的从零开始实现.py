#%% [markdown]
import torch
import torchvision
import numpy as np
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

#%% [markdown]
# 1.读取数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=None, root='Datasets')

#%% [markdown]
# 2.初始化模型参数
num_inputs = 28 * 28
num_outputs = 10

W = torch.tensor(np.random.normal(0,0.01, (num_inputs, num_outputs)), dtype=torch.float)
b = torch.zeros(num_outputs, dtype=torch.float)

W.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

#%% [markdown]
# 3.实现SOFTMAX运算
def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition # 这⾥应⽤了⼴播机制

#%% [markdown]
# 4.定义模型
def net(X):
    return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)

#%% [markdown]
# 5.定义损失函数
def cross_entropy(y_hat, y):
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))

#%% [markdown]
# 6.计算分类准确率
def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()

#%% [markdown]
# 7.训练模型
num_epochs, lr = 5, 0.1
d2l.train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs,batch_size, [W, b], lr)

#%% [markdown]
# 8.预测
X, y = iter(test_iter).next()
true_labels = d2l.get_fashion_mnist_labels(y.numpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels,pred_labels)]
d2l.show_fashion_mnist(X[0:9], titles[0:9])

#%% [markdown]

