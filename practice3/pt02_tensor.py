import torch
import numpy as np
a = np.ones(4)  #返回一个新数组，形状为4，数值默认为1，类型默认为int
b = torch.from_numpy(a) #numpy转为tensor
c = np.ones(4)
np.add(a,2,out=a) #a数组中的数值+2，输出重新赋给a
np.add(a,3,out=c)
print(a)
print(b)
print(c)
x = torch.zeros(1,3,1,1,2,2) #返回数值为0的新数组,从右向左，从内向外建立
print(x)
print(x.size())
y = torch.squeeze(x) #将维数为1的去除，y可等于去掉1的x = torch.zeros(3,2,2)
print(y)
print(y.size())
y1 = torch.squeeze(x,0) #将第0维数据删除（前提：该为维数为1，否则删除失败）
print(y1)
print(y1.size())
y2 = torch.squeeze(x,1) #将第1维数据删除（前提：该为维数为1，否则删除失败，这里维数为3，即失败）
print(y2)
print(y2.size())
z0 = torch.unsqueeze(y,0) #将第0维（注意，这里dim=0时指的是数据为行方向扩）增加维度
print(z0)
print(z0.size())
z1 = torch.squeeze(y,1) #将第1维（注意，这里dim=1时指的是数据为列方向扩）增加维度
print(z1)
print(z1.size())