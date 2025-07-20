import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

batch_size = 64
tranform = transforms.Compose([
    transforms.ToTensor(),  # 将PIL图像或numpy.ndarray转换为pytorch的张量格式Tensor
    transforms.Normalize((0.1307,), (0.3081,))  # 针对 MNIST 数据集的像素分布进行的标准化
])

train_dataset = datasets.MNIST(root='dataset/mnist', train=True, download=False, transform=tranform)
test_dataset = datasets.MNIST(root='dataset/mnist', train=False, download=False, transform=tranform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(kernel_size=2)  # 默认stride等于kernel_size
        self.fc = torch.nn.Linear(320, 10)  #以下有避免手算in_features的方式

    def forward(self, x):
        # Flatten the data (n, 1, 28, 28) -> (n, 784)
        # 在PyTorch中，张量的默认维度顺序是 (B, C, H, W)
        batch_size = x.size(0)
        # 卷积层+relu+池化层
        x = self.pooling(F.relu(self.conv1(x)))
        x = self.pooling(F.relu(self.conv2(x)))
        # 将数据展平
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x

'''
要避免手动计算全连接层的输入维度，可以通过在前向传播中动态推导该值。以下是一个改进后的网络实现，通过自适应扁平化（Adaptive Flattening）自动计算特征维度：
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(kernel_size=2)
        self.fc = None  # 延迟初始化全连接层
        
    def forward(self, x):
        batch_size = x.size(0)
        x = self.pooling(F.relu(self.conv1(x)))
        x = self.pooling(F.relu(self.conv2(x)))
        # 动态计算特征维度
        if self.fc is None:
            # 第一次前向传播时计算并初始化全连接层
            feature_size = x.view(batch_size, -1).size(1)
            
            # 之前的卷积层和池化层是在 __init__ 中初始化的。当整个模型被移动到 GPU 时（例如通过 model.to(device)），这些层会自动跟随模型一起移动。
            # 但 self.fc 是在第一次前向传播时动态创建的。如果不明确指定 .to(x.device)，它默认会在 CPU 上创建，这可能与输入数据 x 的设备不一致。
            self.fc = nn.Linear(feature_size, 10).to(x.device)  #注意要加.to(x.device)

        x = x.view(batch_size, -1)  # 展平为一维向量
        x = self.fc(x)
        return x
'''


model = Net()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} device')
model.to(device)

# 交叉熵损失函数
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

train_losses = []
test_accuracies = []


def train(epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()  # 清零梯度
        outputs = model(inputs)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f'Epoch: {epoch+1}, Loss: {avg_loss}')


def test():
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    test_accuracies.append(accuracy)
    print(f'Accuracy on test set: {accuracy}%')


if __name__ == '__main__':
    for epoch in range(20):
        train(epoch)
        test()

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, label='Test Accuracy')
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.show()

