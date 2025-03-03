import torch

# a = torch.tensor([1.0])
# # 通过requires_grad参数设置是否需要计算梯度
# a.requires_grad = True
# # 或者：a = torch.tensor([1.0], requires_grad=True)

# print(a) #tensor([1.], requires_grad=True)
# print(a.type()) #torch.FloatTensor
# print(a.data) #tensor([1.])
# print(a.data.type()) #torch.FloatTensor
# print(a.item()) #1.0
# print(type(a.item())) #<class 'float'>
# print(a.grad) #None
# print(type(a.grad)) #<class 'NoneType'>

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# w的初始值为1.0，需要计算梯度
w = torch.tensor([1.0], requires_grad=True)

def forward(x):
    return x * w

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

print("Predict (before training)", 4, forward(4).item())

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        # 前向传播，计算损失
        l = loss(x, y)
        # 反向传播，计算梯度 (requires_grad=True)
        # .backward()计算的梯度会累加到.grad属性中
        l.backward()
        print("\tgrad:", x, y, w.grad.item())
        # 更新权重，使用.data以避免自动求导
        #.data是tensor的属性，返回一个新的tensor，这个tensor和原tensor共享data，但是不会记录计算图，不会计算梯度
        w.data = w.data - 0.03 * w.grad.data
        # 梯度清零
        w.grad.data.zero_()
    print("progress:", epoch, l.item())

print("Predict (after training)", 4, forward(4).item())