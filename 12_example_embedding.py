'''
Embedding（嵌入）是一种 把离散的类别数据（比如单词、字母、ID）映射到连续的向量空间 的方法。
输入是整数索引（如单词的编号、字符的编号）
输出是一个稠密向量（embedding vector），它的维度是 embedding_size。
它相当于一个可学习的查找表，每个索引对应一个向量，这些向量会在训练过程中自动更新。
这样做的好处：
降低输入维度：不用用 one-hot 表示（很稀疏，而且维度大），而是用一个低维稠密向量。
保留语义关系：模型可以学到相似的类别有相似的向量（比如 "cat" 和 "dog" 的向量会比 "cat" 和 "car" 更接近）。
'''

import torch

# 1. 参数
num_class = 4  # 类别数
input_size = 4
hidden_size = 8 # 每个时间步隐藏状态的长度（模型记忆容量）
embedding_size = 10 # 嵌入向量的维度
num_layers = 2 # 堆叠的 RNN 层数
batch_size = 1  # 每次处理的序列数
seq_len = 5  # 序列长度

# 2. 数据
index2char = ['e', 'h', 'l', 'o']  # 词典
'''
后面的rnn用了batch_first=True，所以输入的形状是 (batch_size, seq_len, input_size)
因此x_data的形状是 (batch_size, seq_len)，经过embedding后变成 (batch_size, seq_len, embedding_size)，符合rnn的输入要求。

假如是默认模式 batch_first=False，那 RNN 期望的输入形状是：(seq_len, batch_size, input_size)
x_data 就得写成：x_data = [[1], [0], [2], [2], [3]]  # 形状 (seq_len, batch_size)
经过 embedding 就会变成 (5, 1, 10)，才能喂给 RNN
'''
x_data = [[1, 0, 2, 2, 3]]  # (batch_size, seq_len) 用字典中的索引（数字）表示来表示hello
y_data = [3, 1, 2, 3, 2]  # (batch_size * seq_len) 标签：ohlol

# embedding的输入需要LongTensor类型的索引
inputs = torch.LongTensor(x_data)
labels = torch.LongTensor(y_data)

# 3. 模型
class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.emb = torch.nn.Embedding(num_class, embedding_size) # Embedding 权重矩阵 (4 × 10)
        self.rnn = torch.nn.RNN(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, num_class)

    def forward(self, x):
        hidden = torch.zeros(num_layers, x.size(0), hidden_size)  # 初始化隐藏状态
        hidden = hidden.to(device)  # 如果使用GPU，记得把hidden也放到GPU上
        x = self.emb(x) # 转换为嵌入向量 (batch_size, seq_len, embedding_size)
        x, _ = self.rnn(x, hidden) # (batch_size, seq_len, hidden_size)
        x = self.fc(x)  # (batch_size, seq_len, num_class)
        return x.view(-1, num_class) # # 展平为 (batch_size * seq_len, num_class)，方便计算损失

net = Model()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} device')
net.to(device)
inputs, labels = inputs.to(device), labels.to(device)

# 4. 损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = torch.optim.Adam(net.parameters(), lr = 0.05)  # Adam 优化器

# 5. 训练
for epoch in range(15):
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    _, idx = outputs.max(dim=1)  # 返回最大值和最大值的索引。使用 _ 忽略了最大值，只保留了索引 idx
    accuracy = (idx == labels).float().mean()  # 计算准确率
    print('Predicted string: ', ''.join([index2char[x] for x in idx]), end='')
    print(f' | Epoch[{epoch+1}/15] | Loss: {loss.item():.4f} | Accuracy: {accuracy:.4f}')
