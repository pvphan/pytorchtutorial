# https://towardsdatascience.com/linear-regression-with-pytorch-eb6dedead817

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 1, 11)
y = 2 * x

#################################

import torch
from torch.autograd import Variable
class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out

###################################

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

inputDim = 1        # takes variable 'x'
outputDim = 1       # takes variable 'y'
learningRate = 0.01
epochs = 100

model = linearRegression(inputDim, outputDim)
model.to(device)
##### For GPU #######
if use_cuda:
    model.cuda()

####################################

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

####################################

inputs = torch.tensor(x, device=device, dtype=torch.float32).view(-1,1)
labels = torch.tensor(y, device=device, dtype=torch.float32).view(-1,1)

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

    # update parameters
    optimizer.step()

    print('epoch {}, loss {}'.format(epoch, loss.item()))


#####################################

with torch.no_grad(): # we don't need gradients in the testing phase
    predicted = model(inputs).cpu().data.numpy()
    print(predicted)

plt.clf()
plt.plot(x, y, 'go', label='True data', alpha=0.5)
plt.plot(x, predicted, '--', label='Predictions', alpha=0.5)
plt.legend(loc='best')
plt.show()
