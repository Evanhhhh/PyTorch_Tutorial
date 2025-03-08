# SGD: Stochastic Gradient Descent 随机梯度下降
'''
随机梯度下降法和梯度下降法的主要区别在于：
1、损失函数由cost()更改为loss()。cost是计算所有训练数据的损失，loss是计算一个训练数据的损失。对应于源代码则是少了两个for循环。
2、梯度函数gradient()由计算所有训练数据的梯度更改为计算一个训练数据的梯度。
3、本算法中的随机梯度主要是指，每次拿一个训练数据来训练，然后更新梯度参数。本算法中梯度总共更新100(epoch)x3 = 300次。梯度下降法中梯度总共更新100(epoch)次。
'''
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0 #Initial guess of weight.

def forward(x):
    return x * w

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

def gradient(x, y):
    return 2 * x * (x * w - y)

epoch_list = []
cost_list = []
print("Predict (before training)", 4, forward(4))
for epoch in range(100):
    # 每次参数更新时使用了单个样本的梯度，通过逐样本更新参数的形式使用了SGD的思想
    # 但缺乏随机打乱数据的步骤，因此未完全体现SGD的“随机性”
    for x, y in zip(x_data, y_data):
        grad = gradient(x, y)
        w -= 0.02 * grad
        print("\tgrad:", x, y, grad)
        l = loss(x, y)
    print("progress:", epoch, "w=", w, "loss=", l)
    epoch_list.append(epoch)
    cost_list.append(l)

print("Predict (after training)", 4, forward(4))
plt.plot(epoch_list, cost_list)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()