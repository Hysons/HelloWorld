import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable

#Hyper Paramters
#输入的尺寸
input_size = 1
#输出的尺寸
output_size = 1
#训练次数
num_epochs = 1000
#学习率
learning_rate = 0.001

#xtrain生成矩阵数据
x_train = np.array([[2.3], [4.4], [3.7], [6.1], [7.3], [2.1],[5.6], [7.7], [8.7], [4.1],

                    [6.7], [6.1], [7.5], [2.1], [7.2],

                    [5.6], [5.7], [7.7], [3.1]], dtype=np.float32)
#ytrain生产矩阵数据
y_train = np.array([[3.7], [4.76], [4.], [7.1], [8.6], [3.5],[5.4], [7.6], [7.9], [5.3],

                    [7.3], [7.5], [8.5], [3.2], [8.7],

                    [6.4], [6.6], [7.9], [5.3]], dtype=np.float32)

#画图散点图
plt.figure()
#图上设置坐标x和y
plt.scatter(x_train, y_train)
#定义xy轴标签，名称
plt.xlabel('x_train')
plt.ylabel('y_train')
#显示图片
plt.show()

# Linear Regression Model
class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.linear(x)
        return out

model = LinearRegression(input_size, output_size)

# Loss and Optimizer
ctiterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Train the Model
for epoch in range(num_epochs):
    #Convert numpy array to torch Variable
    inputs = Variable(torch.from_numpy(x_train))
    targets = Variable(torch.from_numpy(y_train))

    # Forward + Backward + Optimize
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = ctiterion(outputs, targets)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0:
        print('Epoch [%d/%d], Loss: %.4f'.format(epoch + 1, num_epochs, loss.item()))

# Plot the graph


model.eval()
predicted = model(Variable(torch.from_numpy(x_train))).data.numpy()
plt.plot(x_train, y_train, 'ro')
plt.plot(x_train, predicted, label='predict')
plt.legend()
plt.show()