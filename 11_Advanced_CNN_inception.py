'''
Inception模块是GoogleNet中提出的结构，主要特点是并行使用不同尺寸的卷积核进行特征提取
'''

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root="dataset/mnist", train=True, download=False, transform=transform)
test_dataset = datasets.MNIST(root="dataset/mnist", train=False, download=False, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# Inception 块
class Inception(torch.nn.Module):
    def __init__(self, in_channels):
        # 初始化四个并行分支
        # 分支1: 1x1卷积，捕获点级特征
        # 分支2: 1x1卷积 -> 5x5卷积，捕获较大感受野的特征
        # 分支3: 1x1卷积 -> 3x3卷积 -> 3x3卷积，用更少参数模拟更大感受野
        # 分支4: 平均池化 -> 1x1卷积，保留信息的同时降维
        # 最后将四个分支的输出在通道维度上连接起来，形成更丰富的特征表示。

        super(Inception, self).__init__()
        self.branch1x1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1) #1x1卷积

        self.branch5x5_1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_2 = torch.nn.Conv2d(16, 24, kernel_size=5, padding=2)  #5x5卷积，padding=2保证输入和输出大小一致

        self.brance3x3_1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3_2 = torch.nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3_3 = torch.nn.Conv2d(24, 24, kernel_size=3, padding=1)

        self.branch_pool = torch.nn.Conv2d(in_channels, 24, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3 = self.brance3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        # 该分支先最大池化，再1x1卷积
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        # 将四个分支的输出在通道维度上拼接 (batch_size, C, W, H)
        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        return torch.cat(outputs, dim=1)  # 在通道维度上拼接


# 设计网络
'''
网络结构由以下部分组成：  
卷积层 -> 池化层 -> ReLU激活
第一个Inception模块
卷积层 -> 池化层 -> ReLU激活
第二个Inception模块
全连接层输出
注意第二个卷积层的输入通道是88，这是因为第一个Inception模块输出了16+24+24+24=88个通道。
'''
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(88, 20, kernel_size=5)  #88是Inception块拼接后的输出通道数

        self.incep1 = Inception(in_channels=10)
        self.incep2 = Inception(in_channels=20)

        self.mp = torch.nn.MaxPool2d(2)
        self.l1 = torch.nn.Linear(1408, 10) # 全连接层,不确定第一个参数为多少时，可以先随便写一个，然后运行程序，看报错信息

    def forward(self, x):
        batch_size = x.size(0)
        # 经过第一个卷积层+池化层+激活函数
        x = F.relu(self.mp(self.conv1(x)))
        # 经过第一个Inception块
        x = self.incep1(x)
        # 经过第二个卷积层+池化层+激活函数
        x = F.relu(self.mp(self.conv2(x)))
        # 经过第二个Inception块
        x = self.incep2(x)
        # 将数据展平，方便输入到全连接层
        x = x.view(batch_size, -1)
        x = self.l1(x)
        return x



model = Net()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum=0.5)

def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        input, label = data
        input, label = input.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print("[Epoch %d, Batch %5d] loss: %.3f" % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            image, label = data
            image, label = image.to(device), label.to(device)
            output = model(image)
            _, predicted = torch.max(output.data, 1)  # 返回每一行中最大值的那个元素，以及其索引
            total += label.size(0)
            correct += (predicted == label).sum().item()
    accuracy = 100 * correct / total
    accuracy_list.append(accuracy)
    print('Accuracy on test set: %d %%' % accuracy)


if __name__ == '__main__':
    accuracy_list = []
    for epoch in range(20):
        train(epoch)
        test()
    plt.plot(accuracy_list)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy')
    plt.grid()
    plt.show()