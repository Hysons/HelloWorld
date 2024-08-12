import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

#定义超参数
#输入层大小
input_size = 784
#隐藏层大小
hidden_size = 196
#输出层大小
output_size = 10
#学习率
learning_rate = 0.001
#训练次数
num_epochs = 10
#每次样本大小
batch_size = 100

train_dataset = datasets.MNIST('./datasets', train = True, transform = transforms.ToTensor(), download = True)
test_dataset = datasets.MNIST('./datasets', train = False, transform = transforms.ToTensor(), download= True)

train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)
#记录测试数据的标签信息
test_y=test_dataset.test_labels

class Net(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)
        self.relu = torch.nn.ReLU()

    def forward(self,x):
        print("x的尺寸：",x.shape)
        out = self.fc1(x)
        print("第一次的尺寸：",out.shape)
        out = self.relu(out)
        out = self.fc2(out)
        print("第二次的尺寸：",out.shape)
        return out

#建立前馈神经网络的模型
net = Net(input_size, hidden_size, output_size)
#定义损失函数，二元交叉熵损失
criterion = torch.nn.CrossEntropyLoss()
#输入需要训练的模型参数，并传入学习速率超参数来初始化优化器。这里采用的adam优化器
opitmizer = torch.optim.Adam(net.parameters(), lr = learning_rate)

#训练模型
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.view(-1, 28*28))
        labels = Variable(labels)

        opitmizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs,labels)
        loss.backward()
        opitmizer.step()

        if (i+1) % 100 == 0:
            print(" Epoch: [%d/%d], Step: [%d/%d] , Loss: %.4f"
                  % (epoch+1, num_epochs, i+1, len(train_dataset)// batch_size, loss.item()))

net.eval()
torch.no_grad()
correct = 0
total = 0
Labels = []
Predictions = []
for images, labels in test_loader:
    images = Variable(images.view(-1, 28*28))
    labels = Variable(labels)
    output = net(images)
    _, predicted = torch.max(output.data, 1)
    total += len(labels)
    correct += (predicted == labels).sum()
    #记录每轮的第67个标签
    Labels.append(labels[66])
    Predictions.append(predicted[66])

accuracy = 100 * correct.double() / total
print("Accuracy of the network on the 10000 test images: %d %%" % (accuracy))

#保存模型
torch.save(net.state_dict(), 'feedforward_net.pkl')
#测试输出，取前40个数据进行预测
test_output = net(images[:40])

pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print('prediction number: ' , pred_y)
# print('labels number:     ' , labels[:40].numpy())
print('real number:       ' , test_y[-100:-60].numpy())

#绘图预测10张图片，每个batch的第67个数据
plt_test_y = net(images)
plt_pred_y = torch.max(plt_test_y, 1)[1].data.numpy().squeeze()
for i in range(0,10):
   plt.imshow(test_dataset.train_data[i*batch_size + 66].numpy(), cmap= 'gray')
   plt.title("predicted is %i , the real is %i " % (Predictions[i].numpy(), Labels[i].numpy()))
   plt.show()
#最终结果，10张图预测成功9张，第3张预测失败，实际8，预测0


'''
记录遇见的第一个问题。
最后输出的预测数字与实际数字的差异特别明显。当时的代码为
print('prediction number: ' , pred_y)
print('real number:       ' , test_y[:40].numpy())

pred_y取自test_output，取了前40个数字。而这样test_y取自test_dataset中的原始标签，同样也是取了前40个数字。
理论上来看，两者的数字应该很接近才对。毕竟这版前馈神经网络的准确率在97%，不应该出现几乎完全不同的情况。
于是通过添加log以及反复调试的情况下，发现问题出自上面的for循环中，会依次遍历取出每一轮batch的images和labels数据。
而test_y包含了所有的batch数据，[:40]不仅仅代表前40个数字，同样也包含着取自第一个batch
但pred_y是由之前for循环结束后的images通过神经网络计算而来，这里的images数据取最后一个batch，故[:40]是指的最后一个batch的数据
因此解决方法有两种，一种是test_y[-100:-60]，取最后的数据。另一种是用labels[:40]，与其等效。
'''
