import torch
import torch.nn as nn
import torch.nn.functional as F

class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_ch, bn_size, growth_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(in_ch + i * growth_rate, growth_rate, bn_size)
            self.add_module(f'denselayer{i + 1}', layer)

class _DenseLayer(nn.Sequential):
    def __init__(self, in_ch, growth_rate, bn_size):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(in_ch))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv2d(in_ch, growth_rate * bn_size,
                                           kernel_size=1, stride=1, bias=False))
        self.add_module('norm2', nn.BatchNorm2d(growth_rate * bn_size))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(growth_rate * bn_size, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        return torch.cat((x, new_features), 1)

class _UpBlock(nn.Module):
    def __init__(self, deconv, conv, denseblock):
        super(_UpBlock, self).__init__()
        self.deconvblock = deconv
        self.convblock = conv
        self.densebolck = denseblock

    def forward(self, x1, x2):
        x1 = self.deconvblock(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX//2, diffX - diffX//2, diffY//2, diffY - diffY//2])

        x = torch.cat((x2, x1), dim=1)
        x = self.densebolck(self.convblock(x))

        return x



class DenseUNet(nn.Module):
    def __init__(self, num_classes, bn_size=4):
        super(DenseUNet, self).__init__()
        self.inconv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            _DenseBlock(num_layers=4, in_ch=32, bn_size=bn_size, growth_rate=8)
        )

        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            _DenseBlock(num_layers=4, in_ch=64, bn_size=bn_size, growth_rate=16)
        )

        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            _DenseBlock(num_layers=4, in_ch=128, bn_size=bn_size, growth_rate=32)
        )

        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            _DenseBlock(num_layers=4, in_ch=256, bn_size=bn_size, growth_rate=64)
        )

        self.up1 = _UpBlock(
            nn.Sequential(
                nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            ),
            _DenseBlock(num_layers=4, in_ch=128, bn_size=bn_size, growth_rate=32)
        )

        self.up2 = _UpBlock(
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            ),
            _DenseBlock(num_layers=4, in_ch=64, bn_size=bn_size, growth_rate=16)
        )

        self.up3 = _UpBlock(
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            ),
            _DenseBlock(num_layers=4, in_ch=32, bn_size=bn_size, growth_rate=8)
        )

        self.outconv = nn.Sequential(
            nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0),
            # nn.Softmax(dim=1)
        )


    def forward(self, x):
        x1 = self.inconv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        x = self.down3(x3)

        x = self.up1(x, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        x = self.outconv(x)

        return x


if __name__ == '__main__':
    x = torch.rand((1, 3, 135, 230))
    print('in', x.shape)
    net = DenseUNet(2)
    y = net(x)
    print('out', y.shape)




