import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super()
        self.net = nn.Sequential(
            nn.ConvTranspose2d( 200, 32 * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(32 * 8),
            nn.ReLU(),
            nn.ConvTranspose2d(32 * 8, 32 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32 * 4),
            nn.ReLU(),
            nn.ConvTranspose2d( 32 * 4, 32 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32 * 2),
            nn.ReLU(),
            nn.ConvTranspose2d( 32 * 2, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d( 32, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, tens):
        return self.net(tens)

class Discriminator(nn.Module):
    def __init__(self):
        super()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32 * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32 * 2, 32 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32 * 4),
            nn.LeakyReLU(0.2),
            # state size. (32*4) x 8 x 8
            nn.Conv2d(32 * 4, 32 * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32 * 8),
            nn.LeakyReLU(0.2),
            # state size. (32*8) x 4 x 4
            nn.Conv2d(32 * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, tens):
        return self.net(tens)

def train(netD, netG, loader, loss_func, optimizerD, optimizerG, num_epochs):
    netD.train()
    netG.train()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    for epoch in range(num_epochs):
        for i, data in enumerate(loader, 0):
            netD.zero_grad()
            realtens = data[0].to(device)
            b_size = realtens.size(0)
            label = torch.full((b_size,), 1, dtype=torch.float, device=device) # gen labels
            output = netD(realtens)
            errD_real = loss_func(output, label)
            errD_real.backward() # backprop discriminator fake and real based on label
            noise = torch.randn(b_size, 200, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(0)
            output = netD(fake.detach()).view(-1)
            errD_fake = loss_func(output, label)
            errD_fake.backward() # backprop discriminator fake and real based on label
            errD = errD_real + errD_fake # discriminator error
            optimizerD.step()
            netG.zero_grad()
            label.fill_(1)  
            output = netD(fake).view(-1)
            errG = loss_func(output, label) # generator error
            errG.backward()
            optimizerG.step()
