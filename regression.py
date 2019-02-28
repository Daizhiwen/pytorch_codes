import torch
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1, 1, 200), dim=1)
y = x.pow(2) + 0.2 * torch.rand(x.size())

plt.scatter(x.numpy(), y.numpy())
plt.show()


class Net(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_hidden2, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_features, n_hidden )
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden2)
        self.predict = torch.nn.Linear(n_hidden2, n_output)
        self.init()

    def init(self):
        torch.nn.init.normal(self.hidden.weight, mean=0, std=1)
        torch.nn.init.normal(self.hidden2.weight, mean=0, std=1)
        torch.nn.init.normal(self.predict.weight, mean=0, std=1)

    def forward(self, x):
        x = torch.sigmoid(self.hidden(x))
        x = torch.relu(self.hidden2(x))
        x = self.predict(x)
        return x

net = Net(1, 5, 5, 1)
print(net)

optimizer = torch.optim.SGD(net.parameters(), lr = 0.1)
loss_fn = torch.nn.MSELoss()

for t in range(100000):
    prediction = net(x)
    loss = loss_fn(prediction, y)
    print(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 100 == 0:
        print(loss)
        plt.cla()
        plt.scatter(x.numpy(), y.numpy())
        plt.plot(x.numpy(), prediction.data.numpy(), 'r-', lw=5)
        #plt.text(0.5, 0, 'Loss=%.4f' % loss[0], fontdict={'size':20, 'color': 'red'})

        plt.ioff()
        plt.show()


#print(net(torch.unsqueeze(torch.Tensor([0.2], dtype = torch.float32), dim=1)))