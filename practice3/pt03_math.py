import torch
a = torch.abs(torch.FloatTensor([1, 2, 3,-2])) #定义浮点数数组（1，2，3，-2），并用abs取绝对值
print(a)
b = torch.randn(4) #torch.randn()函数用于生成元素值服从标准正态分布（‌均值为0，‌方差为1）‌的随机数
print(b)
c = torch.add(b,a)
print(c)
d = torch.rand(4) #torch.rand()函数用于生成在区间[0, 1)上均匀分布的随机数
print(d)