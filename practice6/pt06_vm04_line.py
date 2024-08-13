import numpy as np
from visdom import Visdom

viz = Visdom()
#np.linspace()函数用于在指定的起始和结束值之间生成特定数量的等间隔数值。‌这个函数特别适用于需要精确控制元素数量的场景。‌
Y = np.linspace(-5, 5, 10)
#绘制曲线
viz.line(
    #np.column_stack可以将一系列一维数组堆叠成二维数组，‌同时支持不同类型的数据。‌
    Y = np.column_stack((Y * Y, np.sqrt(Y+5))),
    X = np.column_stack((Y, Y)),
    #这里是两组数据，两条线。一个是（Y，Y*Y),一个是（Y,np.sqrt(Y+5))
    opts = dict(markers = False),
)
viz.close()
