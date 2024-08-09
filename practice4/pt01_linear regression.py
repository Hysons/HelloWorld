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
#调用均方差函数MSE做损失函数。第一次运行报错参数模糊不清源于该处方法的括号缺失。
ctiterion = nn.MSELoss()
#输入需要训练的模型参数，并传入学习速率超参数来初始化优化器。
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Train the Model
for epoch in range(num_epochs):
    #Convert numpy array to torch Variable
    inputs = Variable(torch.from_numpy(x_train))
    targets = Variable(torch.from_numpy(y_train))

    # Forward + Backward + Optimize
    #遍历模型的所有参数。通过p.grad.detach_()方法截断反向传播的梯度流，再通过p.grad.zero_()函数将每个参数的梯度值设为0，即上一次的梯度记录被清空。
    optimizer.zero_grad()
    #返回模型前向传播中输出层的值到outputs
    outputs = model(inputs)
    #传入计算值和实际值，调用均方差做损失函数
    loss = ctiterion(outputs, targets)
    #loss会一层层的反向传播计算每个w的梯度值，并保存到该w的.grad属性中。
    loss.backward()
    #执行一次优化步骤，通过梯度下降法来更新参数的值。因为梯度下降是基于梯度的，所以在执行optimizer.step()函数前应先执行loss.backward()函数来计算梯度。
    optimizer.step()

    #每隔5次输出一次损失率
    if (epoch + 1) % 5 == 0:
        print('Epoch [%d/%d], Loss: %.4f'%(epoch + 1, num_epochs, loss.item()))

# Plot the graph

#将模型设置为评估模式，需要关闭一些在训练过程中使用的特性，如Dropout和BatchNorm层的训练模式。在评估模式下，模型将使用训练过程中学到的参数进行前向传播，而不会更新这些参数。
model.eval()
#关闭梯度计算，可以节省内存和计算资源
torch.no_grad()
#输入x轴训练集，输出模型结果作为预测值
predicted = model(Variable(torch.from_numpy(x_train))).data.numpy()
#绘制原始数据，‘ro-’ 表示红色圆点线条。
plt.plot(x_train, y_train, 'ro')
#绘制线性回归曲线
plt.plot(x_train, predicted, label='predict')
#添加图例。如显示标签'predict'
plt.legend()
#图显示出来
plt.show()