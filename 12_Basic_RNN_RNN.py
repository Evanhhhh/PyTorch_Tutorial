'''
使用torch.nn.RNN的简单例子

默认情况下 (batch_first=False):
input.shape = (seq_len, batch_size, input_size)
h_0.shape = (numLayers, batch_size, hidden_size)

output.shape = (seq_len, batch_size, hidden_size)
h_n.shape = (numLayers, batch_size, hidden_size)

如果设置 batch_first=True:
input.shape = (batch_size, seq_len, input_size)
'''

'''
同一 RNN 层中，不同时间步的权重共享
'''

import torch

batch_size = 1
seq_len = 3  # 序列的长度，每个序列包含多少个时间步（或元素） x1, x2, x3...
input_size = 4  #每个时间步（x1）的特征向量维度
hidden_size = 2
num_layers = 1  # 堆叠的 RNN 层的数量

rnn = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
# 如果使用 batch_first=True，交换input_size和 seq_len 的位置
# rnn = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

inputs = torch.randn(seq_len, batch_size, input_size)
# 如果 batch_first=True:
# inputs = torch.randn(batch_size, seq_len, input_size)

hidden = torch.zeros(num_layers, batch_size, hidden_size)

output, hidden = rnn(inputs, hidden)

print("Output size: ", output.shape)
print("Output: ", output)
print("Hidden size: ", hidden.shape)
print("Hidden: ", hidden)