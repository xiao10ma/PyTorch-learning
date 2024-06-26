import torch

'''
RNNCell
'''
# batch_size = 1
# seq_len = 3
# input_size = 4
# hidden_size = 2

# cell = torch.nn.RNNCell(input_size=input_size, hidden_size=hidden_size)

# # (seq, batch, features)
# dataset = torch.randn(seq_len, batch_size, input_size)
# hidden = torch.zeros(batch_size, hidden_size)

# for idx, input in enumerate(dataset):
#     print('=' * 20, idx, '=' * 20)
#     print('Input size:', input.shape)

#     hidden = cell(input, hidden)

#     print('Output size:', hidden.shape)
#     print(hidden)

'''
RNN
'''
# batch_size = 1
# seq_len = 3
# input_size = 4
# hidden_size = 2
# num_layers = 1

# # cell = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size,
# #                     num_layers=num_layers)
# cell = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size,
#                     num_layers=num_layers, batch_first=True)

# # (seqLen, batchSize, inputSize)
# inputs = torch.randn(batch_size, seq_len, input_size)
# hidden = torch.zeros(num_layers, batch_size, hidden_size)

# out, hidden = cell(inputs, hidden)

# print('Output Size: ', out.shape)
# print('Output', out)
# print('Hidden size: ', hidden.shape)
# print('Hidden: ', hidden)

num_class = 4
batch_size = 1
seq_len = 5
input_size = 4
hidden_size = 8
embedding_size = 10
num_layers = 2

idx2char = ['e', 'h', 'l', 'o']
x_data = [1, 0, 2, 2, 3]
y_data = [3, 1, 2, 3, 2]

one_hot_lookup = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
]

x_one_hot = [one_hot_lookup[x] for x in x_data]

inputs = torch.LongTensor(x_one_hot).view(batch_size, -1, input_size)       
labels = torch.LongTensor(y_data)

'''
RNN input/output shape

inputs  [seq_len, batch_size, input_size]   seq_len: 句子长度   batch_size: 一个batch中句子的个数   input_size: embedding length
hidden  [num_layers, batch_size, hidden_size]   num_layers: RNN深度     hidden_size: 隐向量长度

-------------

out     [seq_len, batch_size, hidden_size]  相当于把inputs的最后一维 (input_size) -> (hidden_size)
hidden  [num_layers, batch_size, hidden_size]   最后一个RNN的输出

'''

# class Model(torch.nn.Module):
#     def __init__(self, input_size, hidden_size, batch_size, num_layers=1):
#         super(Model, self).__init__()
#         self.num_layers = num_layers
#         self.batch_size = batch_size
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.rnn = torch.nn.RNN(input_size=self.input_size,
#                                         hidden_size=self.hidden_size,
#                                         num_layers=num_layers)
        
#     def forward(self, input):
#         hidden = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
#         out, _ = self.rnn(input, hidden)
#         return out.view(-1, self.hidden_size)
    
# net = Model(input_size, hidden_size, batch_size)

# criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(net.parameters(), lr=0.05)

# for epoch in range(15):
#     optimizer.zero_grad()
#     outputs = net(inputs)
#     loss = criterion(outputs, labels)
#     loss.backward()
#     optimizer.step()

#     _, idx = outputs.max(dim=1)

#     idx = idx.data.numpy()
#     print('Predicted: ', ''.join([idx2char[x] for x in idx]), end='')
#     print(', Epoch [%d/15] loss = %.4f' % (epoch + 1, loss.item()))


'''
Add embedding
'''



class Model(torch.nn.Module):
    def __init__(self, ):
        super(Model, self).__init__()
        self.emb = torch.nn.Embedding(input_size, embedding_size)
        self.rnn = torch.nn.RNN(input_size=embedding_size,
                                        hidden_size=hidden_size,
                                        num_layers=num_layers,
                                        batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, num_class)

    def forward(self, x):
        hidden = torch.zeros(num_layers, x.size(0), hidden_size)
        x = self.emb(x)
        x, _ = self.rnn(x, hidden)
        x = self.fc(x)
        return x.view(-1, num_class)
    
net = Model()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.05)

for epoch in range(15):
    optimizer.zero_grad()
    outputs = net(torch.LongTensor(x_data).view(batch_size, seq_len))
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    _, idx = outputs.max(dim=1)

    idx = idx.data.numpy()
    print('Predicted: ', ''.join([idx2char[x] for x in idx]), end='')
    print(', Epoch [%d/15] loss = %.4f' % (epoch + 1, loss.item()))