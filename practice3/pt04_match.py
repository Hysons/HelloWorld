import torch

a = torch.randn(2,5) #torch.randn()函数用于生成2行5列的元素值服从标准正态分布（‌均值为0，‌方差为1）‌的随机数
print(a)

b = torch.mean(a) #返回数组元素在给定轴上的平均值。如果未指定dim维度，则返回所有元素的均值
print(b)

c = torch.randn(4,5)
print(c)

d = torch.mean(c,1) #返回数组元素在给1维（即5列）上的平均值（对每行的五个数求平均）。
print(d)