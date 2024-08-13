import numpy as np
from visdom import Visdom

vis=Visdom()

x=np.tile(np.arange(1,101),(100,1))
y=x.transpose()
X=np.exp((((x-50)**2)+((y-50)**2)) / -(20.0 ** 2))
#用于显示等高线图，‌其中X是一个包含等高线数据的二维数组。‌opts参数是一个字典
vis.contour(X=X,opts=dict(colormap="Viridis"))

vis.close()