import numpy as np
from visdom import Visdom

vis = Visdom()

# mesh plot
x =[0,0,1,1,0,0,1,1]
y=[0,1,0,1,0,0,0,1]
z=[1,0,1,1,0,1,1,0]
X=np.c_[x,y,z]
i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2]
j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3]
k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6]
Y = np.c_[i, j, k]
print("X is ",X)
print("Y is ",Y)

#绘制3D网格,opacity代表透明度
vis.mesh(X=X, Y=Y, opts=dict(opacity=0.8))

vis.close()