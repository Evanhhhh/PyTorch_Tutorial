'''
Residual Network (ResNet)
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


# Residual Block
class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        # padding=1 保证输入和输出大小一致
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(x + y)  #残差连接


# ResNet
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=5)

        # residual block
        self.res1 = ResidualBlock(16)
        self.res2 = ResidualBlock(32)

        self.mp = torch.nn.MaxPool2d(2)
        self.l1 = torch.nn.Linear(512, 10)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.mp(F.relu(self.conv1(x)))
        x = self.res1(x)
        x = self.mp(F.relu(self.conv2(x)))
        x = self.res2(x)
        x = x.view(batch_size, -1)
        x = self.l1(x)
        return x


model = Net()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} device')
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print(f'[{epoch + 1}, {batch_idx + 1}] loss: {running_loss / 300:.3f}')
            running_loss = 0.0

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        # 遍历每一个batch的数据
        for data in test_loader:
            images, labels = data  # 获取当前批次的图片和标签
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)  # 累加当前批次的样本数
            correct += (predicted == labels).sum().item()  # 累加当前批次中预测正确的数量
    accuracy = 100 * correct / total
    accuracy_list.append(accuracy)
    print(f'Accuracy on test sef: {accuracy:.2f}%')


def visualize_predictions():
    model.eval()  # 设置为评估模式

    # 获取一批测试数据
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)

    # 获取预测结果
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    # 选择随机的6张图片进行显示
    indices = torch.randperm(len(images))[:6]

    # 创建图表
    fig = plt.figure(figsize=(10, 6))
    for idx, i in enumerate(indices):
        # 添加子图
        ax = fig.add_subplot(2, 3, idx + 1)

        # 获取原始图像并反标准化以便更好地显示
        img = images[i].cpu().squeeze().numpy()

        # 显示图像
        ax.imshow(img, cmap='gray')

        # 设置标题：真实值 vs 预测值
        ax.set_title(f'True: {labels[i].item()}\nLabel: {predicted[i].item()}',
                     color=('green' if predicted[i] == labels[i] else 'red'))
        ax.axis('off')

    plt.tight_layout()
    plt.suptitle('Test the predicted output of random images', fontsize=16)
    plt.subplots_adjust(top=0.85)
    plt.show()


if __name__ == '__main__':
    accuracy_list = []
    for epoch in range(10):
        train(epoch)
        test()

    # 展示一些随机图片的预测结果
    visualize_predictions()

    # Plot the accuracy
    plt.plot(range(1, len(accuracy_list) + 1), accuracy_list)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy')
    plt.grid()
    plt.show()