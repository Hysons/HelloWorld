import numpy as np
from visdom import Visdom
viz = Visdom()
#绘制单张图片
viz.image(
    np.random.rand(3, 256, 256),
    opts=dict(title='single Image', caption='Image title 1'),
)

#绘制多张网格图片
viz.images(
    np.random.rand(20, 3, 256, 256),
    opts=dict(title='network Image', caption='Image title 2'),
)
viz.close()