import math

import numpy as np
from visdom import Visdom

vis = Visdom()

Y = np.linspace(0, 2 * math.pi, 70)
X = np.column_stack((np.sin(Y),np.cos(Y)))
#创建stem图，‌这是一种特殊的图表类型，‌通常用于展示数据点的详细信息，‌包括数据点的值和它们的变化趋势
vis.stem(
    X=X,
    Y=Y,
    opts=dict(legend=['Stem', 'Cosine']),
)

vis.close()