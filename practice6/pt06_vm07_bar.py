import numpy as np
from visdom import Visdom

vis = Visdom()

#绘制单个条形图
vis.bar(
    X = np.random.rand(20)
)

print("1 ",np.random.rand(20))

#绘制堆叠条形图
vis.bar(
    X = np.random.rand(3,5),
    opts= dict(
        stacked = True,
        legend = ['a', 'b', 'c', 'd', 'e'],
        rownames = ['x', 'y', 'z'],
    )
)

print("2 ",np.random.rand(3,5))

#绘制分组条形图
vis.bar(
    X = np.random.rand(20,2),
    opts= dict(
        stacked = False,
        legend = ['True', 'False'],
    )
)

print("3 ",np.random.rand(20,2))
vis.close()