import torch

# Prepare dataset
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])

# Design model using class
class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        # 构造Linear的对象，输入维度是1，输出维度是1
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        # Linear已经实现了魔法方法__call__()，它使类的实例可以像一个函数一样被调用。
        y_pred = self.linear(x)
        return y_pred

model = LinearModel()

# 损失和优化器
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 训练循环：前馈-反馈-优化
for epoch in range(1000):
    y_pred = model(x_data)  # forward: predict
    loss = criterion(y_pred, y_data)  # forward: loss
    print(f'Epoch: {epoch} | Loss: {loss.item()}')
    optimizer.zero_grad()  # 反向传播之前要清零梯度
    loss.backward()  # backward: autograd
    optimizer.step() # 更新参数

# After training
print("w = ", model.linear.weight.item())
print("b = ", model.linear.bias.item())

# Test
x_test = torch.Tensor([[4.0]])  #一行一列
y_test = model(x_test)
print("y_pred = ", y_test.data)