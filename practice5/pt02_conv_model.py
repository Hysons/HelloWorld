import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1dModel(torch.nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride):
        super(Conv1dModel,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv1 = torch.nn.Conv1d(in_channels=self.in_channels,out_channels=self.out_channels,kernel_size=self.kernel_size,stride=self.stride)
        self.conv2 = torch.nn.Conv1d(in_channels=self.out_channels,out_channels=33,kernel_size=3,stride=1)
        self.maxpool1 = torch.nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        print("一维运算")
        x1 = F.relu(self.conv1(x))
        print("第一次卷积后的尺寸 ",x1.shape)
        x2 = F.relu(self.conv2(x1))
        print("第二次卷积后的尺寸 ",x2.shape)
        m1 = self.maxpool1(x2)
        print("第一次池化的尺寸 ",m1.shape)
        return m1

class Conv2dModel(torch.nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride):
        super(Conv2dModel,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv1 = torch.nn.Conv2d(in_channels=self.in_channels,out_channels=self.out_channels,kernel_size=self.kernel_size,stride=self.stride)
        self.conv2 = torch.nn.Conv2d(in_channels=self.out_channels,out_channels=13,kernel_size=3,stride=1)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        print("二维运算")
        x1 = F.relu(self.conv1(x))
        print("第一次卷积后的尺寸 ",x1.shape)
        x2 = F.relu(self.conv2(x1))
        print("第二次卷积后的尺寸 ",x2.shape)
        m1 = self.maxpool1(x2)
        print("第一次池化的尺寸 ",m1.shape)
        return m1

#卷积计算 假设图片的尺寸是 NxN，滤波器尺寸是 FxF，步长为 S，则输出数据的尺寸为：(N-F)/S + 1。所以这里应该是 （22-5）/2 + 1 = 9.
model1 = Conv1dModel(3,20,5,2)
input1 = torch.randn(1,3,22)
output1 = model1(input1)
print(output1.shape)
model2 = Conv2dModel(3,8,4,2)
input2 = torch.randn(1,3,24,16)
output2 = model2(input2)
print(output2.shape)