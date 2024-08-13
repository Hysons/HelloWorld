import numpy as np
from visdom import Visdom

vis = Visdom()

X = np.random.rand(100, 2)
X[:,1] += 2
#箱线图是一种用作显示一组数据分散情况统计图形的图纸，‌主要包含了五个数据节点，‌即最小值、‌下四分位数、‌中位数、‌上四分位数与最大值，‌通过箱线图可以直观地看出数据分布的偏态和尾重情况
vis.boxplot(
    X = X,
    opts=dict(
        legend = ['Men', 'Women'],
    )
)

vis.close()