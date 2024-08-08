import torch

#依次比较每个元素是否相等
a = torch.eq(torch.Tensor([[1,2],[3,4]]), torch.Tensor([[5,6],[7,4]]))
print(a)
#整个数组比较是否相等
b = torch.equal(torch.Tensor([[1,2],[3,4]]), torch.Tensor([[5,2],[3,4]]))
print(b)
# 判断input是否大于等于other，是为ture，否为false。ge(input: Tensor, other: Tensor, *, out: Optional[Tensor] = None)
c = torch.ge(torch.Tensor([[1,2],[3,4]]), torch.Tensor([[5,6],[3,2]]))
print(c)
# 判断input是否大于other，是为ture，否为false。
d = torch.gt(torch.Tensor([[1,2],[3,4]]), torch.Tensor([[5,2],[7,2]]))
print(d)