# https://towardsdatascience.com/linear-regression-with-pytorch-eb6dedead817

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-1, 1, 11)
y = x**2
y[4] = 0.5


#################################

import torch
from torch.autograd import Variable

class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        hiddenSize = 1000
        self.linear1 = torch.nn.Linear(inputSize, hiddenSize)
        self.linear2 = torch.nn.Linear(hiddenSize, hiddenSize)
        self.linear3 = torch.nn.Linear(hiddenSize, outputSize)
        self.relu = torch.nn.ReLU()

        self.nonlinearity = torch.nn.Sequential(
            self.linear1,
            self.relu,
            self.linear2,
            self.relu,
            self.linear3,
        )

    def forward(self, x):
        out = self.nonlinearity(x)
        return out

###################################

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

inputDim = 1        # takes variable 'x'
outputDim = 1       # takes variable 'y'
learningRate = 0.1
epochs = 100

model = linearRegression(inputDim, outputDim)
model.to(device)
# ##### For GPU #######
# if torch.cuda.is_available():
#     model.cuda()

####################################

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

####################################

inputs = torch.tensor(x, device=device, dtype=torch.float32).view(-1,1)
labels = torch.tensor(y, device=device, dtype=torch.float32).view(-1,1)

losses = []

for epoch in range(epochs):

    # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
    optimizer.zero_grad()

    # get output from the model, given the inputs
    outputs = model(inputs)

    # get loss for the predicted output
    loss = criterion(outputs, labels)
    print(loss)

    # get gradients w.r.t to parameters
    loss.backward()

    losses.append(loss.detach().cpu().numpy())

    # update parameters
    optimizer.step()

    print('epoch {}, loss {}'.format(epoch, loss.item()))


#####################################

with torch.no_grad(): # we don't need gradients in the testing phase
    predicted = model(inputs).cpu().data.numpy()
    print(predicted)

print("linear1", model.linear1.weight.data)
print("linear2", model.linear2.weight.data)


plt.clf()
plt.plot(x, y, 'go', label='True data', alpha=0.5)
plt.plot(x, predicted, '--', label='Predictions', alpha=0.5)
plt.legend(loc='best')

plt.figure()
plt.plot(np.array(losses))

plt.show()


