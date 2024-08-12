import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTConvNet(nn.Module):
    def __init__(self):
        super(MNISTConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5,1)
        self.maxpool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(10, 20, 2,2)
        self.maxpool2 = nn.MaxPool2d(2,1)
        self.fc1 = nn.Linear(20*30*30, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x1 = self.maxpool1(F.relu(self.conv1(x)))
        print("第一层卷积池化后尺寸", x1.shape)
        x2 = self.maxpool2(F.relu(self.conv2(x1)))
        print("第二层卷积池化后尺寸", x2.shape)
        #此次必须得将四维张量转为二维张量后，才能够作为全连接层的输入
        x3 = x2.view(x2.size(0), -1)
        f1 = self.fc1(x3)
        print("通过第一层全连接层后尺寸", f1.shape)
        f2 = F.relu(self.fc2(f1))
        print("通过第二层全连接层后尺寸", f2.shape)
        return f2

net = MNISTConvNet()
print(net)
x = torch.randn(1,1, 128, 128)
print(net(x))

'''
记录写这个类遇见的一个问题。
在加入了全连接层的方法后，运行始终报错。mat1 and mat2 shapes cannot be multiplied (600x30 and 18000x64)
当时不理解原因，x2的尺寸torch.Size([1, 20, 30, 30])，而fc1的线性输入通道设置的20*30*30，是与之相对应的，为何会报错。
在网络上搜了一部分发现写法也是如此，故而愈发迷茫
后在仔细看了PyTorch的nn.Linear()详解后，发现作为全连接的输出，与卷积层是有差异的。2D卷积层的输入是四维张量，但是全连接层的输入确是二维张量。
故在输入之前，应当对四维变量的维度进行调整，将为2维后再输入。即添加了x3 = x2.view(x2.size(0), -1)这行代码，顺利解决问题。
'''