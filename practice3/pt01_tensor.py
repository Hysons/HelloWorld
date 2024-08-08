import torch
#定义5行7列的张量
x = torch.Tensor(5,7)
y = torch.randn(5,7)
z = x.numpy()
print(x)
print(y)
print(z)
print(x+y)
print(torch.add(x,y))