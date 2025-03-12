import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

batch_size = 64
# 转换PIL图像为Tensor
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='dataset/mnist', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/mnist', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(784, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)

    def forward(self, x):
        # Flatten the data (n, 1, 28, 28) -> (n, 784)
        x = x.view(-1, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        # 最后一层不需要激活函数，因为在交叉熵损失函数中已经包含了softmax
        return self.l5(x)


model = Net().to(device)
criterion = torch.nn.CrossEntropyLoss()
# momentum 动量参数，加速SGD在相关方向上的移动，抑制震荡
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

train_losses = []
test_accuracies = []


def train(epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # if (batch_idx + 1) % 300 == 0:
        #     print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
        #     running_loss = 0.0
    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f'Epoch: {epoch+1}, Loss: {avg_loss:.4f}')


def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # 取概率最大的下标。_表示忽略返回值（即概率值），predicted表示预测结果，dim=1表示在行上取
            # The torch.max function returns a named tuple (values, indices)
            _, predicted = torch.max(outputs.data, dim=1)
            # labels.size(0)表示batch_size
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    test_accuracies.append(accuracy)
    print(f'Accuracy on test set: {accuracy:.4f}%')


# 可视化部分预测结果
def visualize_predictions():
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images[:10].to(device), labels[:10].to(device)
    outputs = model(images)
    _, predictions = torch.max(outputs, 1)
    images = images.cpu().numpy()
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    for i, ax in enumerate(axes.flat):
        img = images[i].squeeze()
        ax.imshow(img, cmap='gray')
        ax.set_title(f'Pred: {predictions[i].item()}')
        ax.axis('off')
    plt.show()

if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()

    # 绘制损失变化
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.show()

    # 绘制测试集准确率
    plt.figure(figsize=(10, 5))
    plt.plot(test_accuracies, label='Test Accuracy', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy Curve')
    plt.legend()
    plt.show()

    # 显示部分测试样本预测结果
    visualize_predictions()
