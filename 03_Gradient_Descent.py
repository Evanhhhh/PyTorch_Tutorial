# 1. 梯度下降 Gradient Descent
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0 #Initial guess of weight.

# Define the model: Linear Model
def forward(x):
    return x * w

# 梯度下降每次迭代需计算整个训练集的平均梯度，即使用所有样本的梯度均值更新参数
def cost(xs, ys):
    cost  = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        cost += (y_pred - y) ** 2
    return cost / len(xs)

def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w - y)  #计算得到的梯度
    return grad / len(xs)

epoch_list = []
cost_list = []
print("Predict (before training)", 4, forward(4))
for epoch in range(100):
    cost_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    w -= 0.05 * grad_val  #学习率为0.05
    print("Epoch:", epoch, "w=", w, "loss=", cost_val)
    epoch_list.append(epoch)
    cost_list.append(cost_val)

print("Predict (after training)", 4, forward(4))
plt.plot(epoch_list, cost_list)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()