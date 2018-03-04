import torch.nn as nn
from torch.autograd import Variable


class CNN(nn.Module):
    def __init__(self, in_shape, n_classes):
        super().__init__()
        c, w, h = in_shape
        pool_layers = 2
        fc_h = int(h / 2**pool_layers)
        fc_w = int(w / 2**pool_layers)
        self.features = nn.Sequential(
            *conv_bn_relu(c, 16, kernel_size=1, stride=1, padding=0),
            *conv_bn_relu(16, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2), #size/2
            *conv_bn_relu(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2), #size/2
        )
        self.classifier = nn.Sequential(
            *linear_bn_relu_drop(64 * fc_h * fc_w, 128, dropout=0.5),
            nn.Linear(128, n_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def conv_bn_relu(in_chans, out_chans, kernel_size=3, stride=1,
                 padding=1, bias=False):
    return [
        nn.Conv2d(in_chans, out_chans, kernel_size=kernel_size,
                  stride=stride, padding=padding, bias=bias),
        nn.BatchNorm2d(out_chans),
        nn.ReLU(inplace=True),
    ]

def linear_bn_relu_drop(in_chans, out_chans, dropout=0.5, bias=False):
    layers = [
        nn.Linear(in_chans, out_chans, bias=bias),
        nn.BatchNorm1d(out_chans),
        nn.ReLU(inplace=True)
    ]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    return layers

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
