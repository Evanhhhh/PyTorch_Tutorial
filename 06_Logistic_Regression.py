import torch
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[0], [0], [1]])

class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        # 加了sigmoid函数
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

model = LogisticRegressionModel()

# 二元交叉熵损失
criterion = torch.nn.BCELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(f'Epoch: {epoch} | Loss: {loss.item()}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("w = ", model.linear.weight.item())
print("b = ", model.linear.bias.item())

# x_test = torch.Tensor([[4.0]])
# y_test = model(x_test)
# print("y_pred = ", y_test.data)

# 生成一个包含200个点的等差数列，范围从0到10。形如 [0.0, 0.05, 0.1, ..., 9.95, 10.0] 的一维数组。
x = np.linspace(0, 10, 200)
# 调整张量形状为 (200, 1)，即200行1列的二维张量。
x_t = torch.Tensor(x).view((200, 1))
y_t = model(x_t)
y = y_t.data.numpy()  # Matplotlib绘图需要NumPy数组或Python原生数据类型。
plt.plot(x, y)
plt.plot([0, 10], [0.5, 0.5], c = 'r')  # 绘制y=0.5的红色直线
plt.xlabel('Hours')
plt.ylabel('Probability of Pass')
plt.grid()
plt.show()