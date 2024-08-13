import numpy as np
from visdom import Visdom

vis = Visdom()

#热力图，可以用颜色变化来反映二维矩阵或表格中的数据信息，可以直观地将数据值的大小以定义的颜色深浅表示出来。
vis.heatmap(
#np.outer用于计算两个数组的外积。‌这个函数将两个一维数组作为输入，‌并返回它们的外积结果。‌这里将返回一个5x10的二维数组，‌其中每个元素是对应位置上的两个数组元素的乘积。
    X = np.outer(np.arange(1,6),np.arange(1,11)),
    opts=dict(
        title='Heatmap',
        columnnames = ['a','b','c','d','e','f','g','h','i','j'],
        rownames = ['y1','y2','y3','y4','y5'],
        colormap = 'rainbow',
    )
)

vis.close()