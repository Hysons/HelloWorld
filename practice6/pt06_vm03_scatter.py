import numpy as np
from visdom import Visdom

vis = Visdom()

#绘制散点图
vis.scatter(
    # 参数Y用来指定点的分布，win指定图像的窗口名称，env指定图像所在的环境，opts通过字典来指定一些样式
    # 生成一个 255x2 的数组，包含随机数.服从标准正态分布（均值为0，标准差为1）的随机数的函数.rand取值是[0,1)，randn取值是理论上（负无穷，正无穷），实际在0附近徘徊
    X = np.random.rand(255,2),
    Y = (np.random.randn(255) > 0) + 1,
    win = "3D scatter",
    env = "main",
    opts=dict(
        markersize = 10,
        markercolor = np.floor(np.random.random((2,3))*255),
    legend=['Men','Women']
    ),
)

vis.close()

