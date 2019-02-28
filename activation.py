import torch
#import torch.nn.functional as F
import matplotlib.pyplot as plt

#fake data
x = torch.linspace(-5, 5, 200)
x_np = x.numpy()

y_relu = torch.relu(x)
y_sigmoid = torch.sigmoid(x)

y_relu_np = y_relu.numpy()
y_sigmoid_np = y_sigmoid.numpy()

plt.figure(1)
plt.subplot(211)
plt.plot(x_np, y_relu_np)
plt.subplot(212)
plt.plot(x_np, y_sigmoid_np)
plt.show()

