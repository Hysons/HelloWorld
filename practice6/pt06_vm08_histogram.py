import numpy as np
from visdom import Visdom

vis = Visdom()

#绘制直方图
vis.histogram(
    X=np.random.rand(100),
    opts=dict(
        #设置了直方图的柱状数量为20
        numbins=20,
    )
)

vis.close()

#纵坐标为个数
