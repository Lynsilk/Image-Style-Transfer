import torch
from torch import nn

def build_D(in_channels, mid_channels):
    net = Discriminator(in_channels, mid_channels)
    return net

class Discriminator(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(Discriminator, self).__init__()
        network = [
                    nn.Conv2d(in_channels, mid_channels, kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(mid_channels, mid_channels*2, kernel_size=4, stride=2, padding=1),
                    nn.InstanceNorm2d(mid_channels * 2),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(mid_channels*2, mid_channels*4, kernel_size=4, stride=2, padding=1),
                    nn.InstanceNorm2d(mid_channels * 4),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(mid_channels*4, mid_channels*8, kernel_size=4, stride=1, padding=1),
                    nn.InstanceNorm2d(mid_channels * 8),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(mid_channels*8, 1, kernel_size=4, stride=1, padding=1),
        ]
        self.model = nn.Sequential(*network)

    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    model = Discriminator(3, 64)
    print(model)
    x = torch.zeros((2, 3, 300, 300))
    y = model(x)
    print(y.size())