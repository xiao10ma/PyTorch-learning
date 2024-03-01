import torch
import numpy as np

# 1. prepare dataset
xy = np.loadtxt('diabetes.csv.gz', delimiter=',', dtype=np.float32)
x_data = torch.from_numpy(xy[:, :-1])
y_data = torch.from_numpy(xy[:, -1])

print('data')

# 2. design model
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x
    
model = Model()

# 3. construct loss and optimizer
criterion = torch.nn.BCELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 4. training cycle
for epoch in range(10000):
    # forward
    y_pred = model(x_data)
    loss = criterion(y_pred.view(-1), y_data)
    print(epoch, loss.item())

    # backward
    optimizer.zero_grad()
    loss.backward()

    # update
    optimizer.step()

