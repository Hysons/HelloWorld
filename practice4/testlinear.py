#对照pt01_linear regression，又敲了一遍，将数据集x和y坐标对调了一下

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable

inputsize = 1
outputsize = 1
num_epochs = 1000
learning_rate = 0.001

x_train = np.array([[3.7], [4.76], [4.], [7.1], [8.6], [3.5],[5.4], [7.6], [7.9], [5.3],

                    [7.3], [7.5], [8.5], [3.2], [8.7],

                    [6.4], [6.6], [7.9], [5.3]], dtype = np.float32)

y_train = np.array([[2.3], [4.4], [3.7], [6.1], [7.3], [2.1],[5.6], [7.7], [8.7], [4.1],

                    [6.7], [6.1], [7.5], [2.1], [7.2],

                    [5.6], [5.7], [7.7], [3.1]], dtype = np.float32)

plt.figure()
plt.scatter(x_train,y_train)
plt.xlabel("x_train")
plt.ylabel("y_train")
plt.show()

class LinearRegression(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        output = self.linear(x)
        return output

model = LinearRegression(inputsize, outputsize)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
for epoch in range(num_epochs):
    inputs = Variable(torch.from_numpy(x_train))
    targets = Variable(torch.from_numpy(y_train))

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0:
        print("Epoch [%d/%d], Loss: %.4f" % (epoch + 1, num_epochs, loss.item()))

model.eval()
torch.no_grad()
predicted = model(Variable(torch.from_numpy(x_train))).data.numpy()
plt.plot(x_train,y_train,"ro")
plt.plot(x_train,predicted,label="predicted")
plt.legend()
plt.show()