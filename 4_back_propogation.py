# import torch

# x_data = [1.0, 2.0, 3.0]
# y_data = [2.0, 4.0, 6.0]

# w = torch.Tensor([1.0])
# w.requires_grad = True

# def forward(x):
#     return x * w

# def loss(x, y):
#     y_pred = forward(x)
#     return (y - y_pred) ** 2

# print('predict (befor training)', 4, forward(4).item())

# for epoch in range(100):
#     for x, y in zip(x_data, y_data):
#         # 计算误差
#         l = loss(x, y)
#         # 反向传播
#         l.backward()
#         print('\tgrad:', x, y, w.grad.item())
#         # 更新，必须要是.data
#         w.data = w.data - 0.01 * w.grad.data
#         # 梯度清零
#         w.grad.data.zero_()

# print("predict (after training)", 4, forward(4).item())

import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = torch.Tensor([1.0, 1.0, 1.0])
w.requires_grad = True

def forward(x):
    return w[0] * (x ** 2) + w[1] * x + w[2]

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

print('predict (before training)', 4, forward(4).item())

for epoch in range(10000):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward()

        # print('\t grad:', x, y, w.grad.data)

        w.data = w.data - 0.01 * w.grad.data
        w.grad.data.zero_()

print('predict (after training)', 4, forward(4).item())
print(w[0].item(), w[1].item(), w[2].item())