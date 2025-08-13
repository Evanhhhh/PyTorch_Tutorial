'''
序列变换：“hello”-> “ohlol”
使用RNN
'''

import torch

# 1. 参数
seq_len = 5
input_size = 4
hidden_size = 4
batch_size = 1

# 2. 数据
index2char = ['e', 'h', 'l', 'o']  # 词典
x_data = [1, 0, 2, 2, 3]  # hello
y_data = [3, 1, 2, 3, 2]  # 标签：ohlol

# 用来将 x_data 转换为 one-hot vector 的参照表
'''
将 x_data 转换为 one-hot 向量，是为了把字符（离散数据）变成神经网络能理解的数值输入形式，并保留其“身份”。
如果直接喂入 0、1、2、3，模型可能会误解它们之间有数值远近的意义，比如认为“l 比 h 多1”。
'''
one_hot_lookup = [[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]
x_one_hot = [one_hot_lookup[x] for x in x_data]

inputs = torch.Tensor(x_one_hot).view(seq_len, batch_size, input_size)
# CrossEntropyLoss 要求标签必须是整数索引形式的 LongTensor，因为它要根据索引选择类别并计算损失。
labels = torch.LongTensor(y_data)  # 标签需要是 LongTensor


# 3. 模型
class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, num_layers=1):
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.rnn = torch.nn.RNN(input_size, hidden_size, num_layers)

    def forward(self, input):
        # 1. 初始化隐藏状态
        h_0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        h_0 = h_0.to(device)  # 如果使用GPU，记得把h_0也放到GPU上
        # 2. 前向传播
        output, h_n = self.rnn(input, h_0)
        '''
        将输出的三维张量 (seq_len, batch_size, hidden_size) 转换为二维张量 (seq_len * batch_size, hidden_size)
        因为CrossEntropyLoss 的输入要求：
        loss = criterion(output, target)
        output 形状：(N, C)，表示有 N 个样本，每个样本有 C 个类别的得分
        target 形状：(N, )，表示 N 个样本的正确类别索引（0 到 C-1 之间的整数）
        
        在这个例子中，hidden_size 必须等于类别数（即 4），否则会导致 CrossEntropyLoss 的输入和标签不匹配；或者在 RNN 输出后添加一个全连接层，将输出维度映射到正确的类别数
        '''
        return output.view(-1, self.hidden_size)  # 展平为 (seq_len * batch_size, hidden_size)


net = Model(input_size, hidden_size, batch_size)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} device')
net.to(device)
inputs, labels = inputs.to(device), labels.to(device)


# 4. 损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.05)

# 5. 训练
for epoch in range(15):
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    _, idx = outputs.max(dim=1)  # 返回最大值和最大值的索引。使用 _ 忽略了最大值，只保留了索引 idx
    print(f"Predicted string: {''.join([index2char[i] for i in idx])} | Epoch[{epoch+1}/15] | Loss: {loss.item()}")