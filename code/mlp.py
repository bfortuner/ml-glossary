import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

# import torchvision module to handle image manipulation
import torchvision
import torchvision.transforms as transforms

# calculate train time, writing train data to files etc.
import time
import pandas as pd
import json


class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        # define layers
        self.fc1 = nn.Linear(in_features=28*28, out_features=500)
        self.fc2 = nn.Linear(in_features=500, out_features=200)
        self.fc3 = nn.Linear(in_features=200, out_features=100)
        self.out = nn.Linear(in_features=100, out_features=10)


    def forward(self, t):
        # fc1  make input 1 dimentional
        t = t.view(-1,28*28)
        t = self.fc1(t)
        t = F.relu(t)
        # fc2
        t = self.fc2(t)
        t = F.relu(t)
        # fc3
        t = self.fc3(t)
        t = F.relu(t)
        # output
        t = self.out(t)
        return t

def train(net, loader, loss_func, optimizer):
    net.train()
    n_batches = len(loader)
    for inputs, targets in loader:
        inputs = Variable(inputs)
        targets = Variable(targets)

        output = net(inputs)
        loss = loss_func(output, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
         # print statistics
    running_loss = loss.item()
    print('Training loss: %.3f' %( running_loss))

def main():
    train_set = torchvision.datasets.FashionMNIST(
        root = './FMNIST',
        train = True,
        download = False,
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
    )
    mlp = MLP()
    loader = torch.utils.data.DataLoader(train_set, batch_size = 1000)
    optimizer = optim.Adam(mlp.parameters(), lr=0.01)
    loss_func=nn.CrossEntropyLoss()
    for i in range(0,15):
        train(mlp,loader,loss_func,optimizer)
    print("Finished Training")
    torch.save(mlp.state_dict(), "./mlpmodel.pt")
    test_set = torchvision.datasets.FashionMNIST(
        root = './FMNIST',
        train = False,
        download = False,
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
    )
    testloader = torch.utils.data.DataLoader(test_set, batch_size=4)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = mlp(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))


main()
