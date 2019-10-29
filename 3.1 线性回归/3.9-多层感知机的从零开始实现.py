#%%
import torch
import numpy as np
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l


#%% [markdown]
# 1.读取数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=None, root='Datasets')

#%% [markdown]
# 2.定义模型参数
num_inputs = 28 * 28
num_outputs = 10
num_hiddens = 256

W1 = torch.tensor(np.random.normal(0,0.01, (num_inputs, num_hiddens)), dtype=torch.float)
b1 = torch.zeros(num_hiddens, dtype=torch.float)

W2 = torch.tensor(np.random.normal(0,0.01, (num_hiddens, num_outputs)), dtype=torch.float)
b2 = torch.zeros(num_outputs, dtype=torch.float)

params = [W1, b1, W2, b2]
for param in params:
    param.requires_grad_(requires_grad=True)

#%% [markdown]
# 3.定义激活函数
def relu(X):
    return torch.max(input=X, other=torch.tensor(0.0))

#%% [markdown]
# 4.定义模型
def net(X):
    X = X.view((-1, num_inputs))
    H = relu(torch.matmul(X, W1) + b1)
    return torch.matmul(H, W2) + b2

#%% [markdown]
# 5.定义损失函数
loss = torch.nn.CrossEntropyLoss()

#%% [markdown]
# 6.训练模型
num_epochs, lr = 5, 100.0
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)

#%% [markdown]
# 9.预测
X, y = iter(test_iter).next()
true_labels = d2l.get_fashion_mnist_labels(y.numpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels,pred_labels)]
d2l.show_fashion_mnist(X[0:9], titles[0:9])

# %%
