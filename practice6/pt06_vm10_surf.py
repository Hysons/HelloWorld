import numpy as np
from visdom import Visdom

vis = Visdom()

x = np.tile(np.arange(1, 101), (100,1))
print(x)
y = x.transpose()
print(y)
#留意括号的书写，之前没能成功绘图就因为i少了一对括号，导致计算数据错误
#((x - 50) ** 2) + ((y - 50) ** 2) 是计算点 (x, y) 到中心点 (50, 50) 的欧几里得距离的平方。‌20.0 是标准差
#np.exp 是计算以自然常数e为底的指数函数。‌
X = np.exp((((x - 50) ** 2) + ((y - 50) ** 2)) / -(20.0 ** 2))
print(X)

#绘制曲面图
vis.surf(X=X,
         opts=dict(colormap='Hot'))

vis.close()