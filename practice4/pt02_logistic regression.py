import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

#设置超参数
#设置图片尺寸 784=28*28
picture_size = 784
#输出的图片种类，10种，0-9
num_classes = 10
#训练次数
num_epochs = 20
#每次样本batch的数量
batch_size = 100
#学习率
learning_rate = 0.002

#获取训练集和测试集数据dateset（包含图像和标签），train代表是否为训练用。transforms.ToTensor是将其转换为tensor类型，便于数据处理，download代表下载
train_dataset = datasets.MNIST(root = './dataset',train=True,transform = transforms.ToTensor(),download = True)
test_dataset = datasets.MNIST(root = './dateset',train=False,transform = transforms.ToTensor(),download = True)

#loader代表数据集加载器，其中shuffle参数代表是否乱序
train_loader = DataLoader(dataset = train_dataset,batch_size = batch_size,shuffle = True)
test_loader = DataLoader(dataset = test_dataset,batch_size = batch_size,shuffle = False)

#定义逻辑回归类
class LogisticRegression(torch.nn.Module):
    def __init__(self, picture_size, num_classes):
        super(LogisticRegression,self).__init__()
        self.linear = torch.nn.Linear(picture_size,num_classes)

    def forward(self, x):
        output = self.linear(x)
        return output

#建立逻辑回归的模型
model = LogisticRegression(picture_size,num_classes)
#定义损失函数，二元交叉熵损失
criterion = torch.nn.CrossEntropyLoss()
#输入需要训练的模型参数，并传入学习速率超参数来初始化优化器。
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

reault = []
#训练模型
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.view(-1, 28 * 28))
        labels = Variable(labels)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        #每读取100次打印
        if (i+1) % 100 == 0:
            print("Epoch: [%d/%d], Step: [%d/%d],Loss: %.4f"
                  % (epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.item()))

    reault.append(float(loss))

fig = plt.figure()
plt.plot(range(len(reault)), reault)
fig.show()

#测试模型，设为评估
model.eval()
#去掉梯度计算
torch.no_grad()

correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images.view(-1,28 * 28)).float()
    labels = Variable(labels)
    outputs = model(images)
    #输出第一个代表最大值，第二个代表索引。这里无需采用最大值，设_即可。如果dim=0，按第一维度行上寻找最大值。
    #这里为1，按第二维度列虚招最大值。这行语句作用是从模型的输出中找到每个样本的最大类别概率及其对应的类别索引
    _, predicted = torch.max(outputs.data, 1)
    #计算标签总数
    total += len(labels)
    #预测正确的标签数量
    correct += (labels == predicted).sum()
#准确率
accuracy = correct.double() / total
print('Accuracy of the network on the 10000 test images: %d %%' % (accuracy * 100))

#模型保存
torch.save(model.state_dict(), 'logistic_regression_model.pkl')