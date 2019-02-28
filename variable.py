import torch
#from torch.autograd import variable

data = [1, 2, 4, 5]
tensor = torch.tensor(data, dtype = torch.float32, requires_grad=True)
#Variable = variable(tensor)

print(tensor)
a = torch.mean(tensor.dot(tensor))
a.backward()

print(tensor.grad)