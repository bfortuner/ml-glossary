
import torch.nn as nn
from torch.autograd import Variable


class Autoencoder(nn.Module):
    def __init__(self, in_shape):
        super().__init__()
        c,h,w = in_shape
        self.encoder = nn.Sequential(
            nn.Linear(c*h*w, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, c*h*w),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs,c,h,w = x.size()
        x = x.view(bs, -1)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(bs, c, h, w)
        return x


class ConvAutoencoder(nn.Module):
    def __init__(self, in_shape):
        super().__init__()
        c,h,w = in_shape
        self.encoder = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=3, stride=1, padding=1),  # b, 16, 32, 32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # b, 16, 16, 16
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),  # b, 8, 16, 16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # b, 8, 8, 8
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=0),  # 16, 17, 17
            nn.ReLU(),
            nn.ConvTranspose2d(16, c, kernel_size=3, stride=2, padding=1),  # 3, 33, 33
            CenterCrop(h, w), # 3, 32, 32
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train(net, loader, loss_func, optimizer):
    net.train()
    for inputs, _ in loader:
        inputs = Variable(inputs)

        output = net(inputs)
        loss = loss_func(output, inputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
