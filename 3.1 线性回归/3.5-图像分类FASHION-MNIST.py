#%%
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import sys
sys.path.append("..") # 为了导⼊上层⽬录的d2lzh_pytorch
import d2lzh_pytorch as d2l

# %%
mnist_train = torchvision.datasets.FashionMNIST(root='Datasets/FashionMNIST', train=True, download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='Datasets/FashionMNIST', train=False, download=True, transform=transforms.ToTensor())
print(type(mnist_train))
print(len(mnist_train), len(mnist_test))

# %%
feature, label = mnist_train[0]
print(feature.shape, label) # Channel x Height X Width
X, y = [], []
for i in range(10):
X.append(mnist_train[i][0])
y.append(mnist_train[i][1])
show_fashion_mnist(X, get_fashion_mnist_labels(y))