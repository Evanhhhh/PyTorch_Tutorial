import numpy as np
import torch
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载数据并移动到 GPU
xy = np.loadtxt("diabetes.csv", delimiter=",", dtype=np.float32)
x_data = torch.from_numpy(xy[:, 0:-1]).to(device)
y_data = torch.from_numpy(xy[:, [-1]]).to(device)


# 定义模型
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


# 实例化模型并移动到 GPU
model = Model().to(device)

# 定义损失函数和优化器
criterion = torch.nn.BCELoss(reduction='mean')  # BCELoss 无参数，无需移动
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# 训练过程
epoch_list, loss_list, acc_list = [], [], []
for epoch in range(100000):
    # 前向传播
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)

    # 记录损失和周期
    epoch_list.append(epoch)
    loss_list.append(loss.item())

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 每隔 10000 周期计算准确率（禁用梯度计算）
    if epoch % 10000 == 9999:
        model.eval()  # 切换到评估模式（影响 Dropout/BatchNorm 层）
        with torch.no_grad():  # 禁用梯度计算
            y_pred_label = torch.where(
                y_pred >= 0.5,
                torch.tensor([1.0], device=device),  # 确保张量在 GPU 上
                torch.tensor([0.0], device=device)
            )
            acc = torch.eq(y_pred_label, y_data).sum().item() / y_data.size(0)
            acc_list.append(acc)
            print(f"Epoch: {epoch} | Loss: {loss.item()} | Accuracy: {acc}")
        model.train()  # 切换回训练模式

# 绘图
fig, ax1 = plt.subplots()
ax1.plot(epoch_list, loss_list, 'b-')
ax1.set_xlabel('epoch')
ax1.set_ylabel('loss', color='b')
ax1.tick_params('y', colors='b')

ax2 = ax1.twinx()
ax2.plot(epoch_list[9999::10000], acc_list, 'r-')
ax2.set_ylabel('accuracy', color='r')
ax2.tick_params('y', colors='r')
plt.show()