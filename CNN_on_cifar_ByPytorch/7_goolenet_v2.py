import torch
import torch.nn as nn

'''
主要的改进：
1. 加入BN
2. 5x5的卷积核变成两个3x3的卷积核
'''
def ConvBNReLU(in_channels, out_channels, kernel_size):

    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels1, out_channels2red, out_channels2, out_channels3red, out_channels3, out_channels4):
        super(InceptionModule, self).__init__()

        self.branch1_conv = ConvBNReLU(in_channels=in_channels, out_channels=out_channels1, kernel_size=1)

        self.branch2_conv1 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels2red, kernel_size=1)
        self.branch2_conv2 = ConvBNReLU(in_channels=out_channels2red, out_channels=out_channels2, kernel_size=3)

        self.branch3_conv1 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels3red, kernel_size=1)
        self.branch3_conv2 = ConvBNReLU(in_channels=out_channels3red, out_channels=out_channels3red, kernel_size=3)
        self.branch3_conv3 = ConvBNReLU(in_channels=out_channels3red, out_channels=out_channels3, kernel_size=3)

        self.branch4_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch4_conv1 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels4, kernel_size=1)

    def forward(self, x):
        out1 = self.branch1_conv(x)
        out2 = self.branch2_conv2(self.branch2_conv1(x))
        out3 = self.branch3_conv3(self.branch3_conv2(self.branch3_conv1(x)))
        out4 = self.branch4_conv1(self.branch4_pool(x))

        # 输出连接
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out

class InceptionAux(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionAux, self).__init__()

        self.auxiliary_avgpool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.auxiliary_conv1 = ConvBNReLU(in_channels=in_channels, out_channels=128, kernel_size=1)
        self.auxiliary_linear1 = nn.Linear(in_features=128 * 4 * 4, out_features=1024)
        self.auxiliary_relu = nn.ReLU(inplace=True)
        self.auxiliary_dropout = nn.Dropout(p=0.7)
        self.auxiliary_linear2 = nn.Linear(in_features=1024, out_features=out_channels)

    def forward(self, x):
        x = self.auxiliary_conv1(self.auxiliary_avgpool(x))
        x = x.view(x.size(0), -1)
        x = self.auxiliary_relu(self.auxiliary_linear1(x))
        out = self.auxiliary_linear2(self.auxiliary_dropout(x))

        return out

class GoogLeNet_v2(nn.Module):
    def __init__(self, num_class=1000, stage='train'):
        super(GoogLeNet_v2, self).__init__()
        self.stage = stage

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.block3 = nn.Sequential(
            InceptionModule(in_channels=192, out_channels1=64, out_channels2red=96, out_channels2=128,
                            out_channels3red=16, out_channels3=32, out_channels4=32),
            InceptionModule(in_channels=256, out_channels1=128, out_channels2red=128, out_channels2=192,
                            out_channels3red=32, out_channels3=96, out_channels4=64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.block4_1 = InceptionModule(in_channels=480, out_channels1=192, out_channels2red=96, out_channels2=208,
                                        out_channels3red=16, out_channels3=48, out_channels4=64)

        if self.stage == 'train':
            self.aux_logits1 = InceptionAux(in_channels=512, out_channels=num_class)

        self.block4_2 = nn.Sequential(
            InceptionModule(in_channels=512, out_channels1=160, out_channels2red=112, out_channels2=224,
                            out_channels3red=24, out_channels3=64, out_channels4=64),
            InceptionModule(in_channels=512, out_channels1=128, out_channels2red=128, out_channels2=256,
                            out_channels3red=24, out_channels3=64, out_channels4=64),
            InceptionModule(in_channels=512, out_channels1=112, out_channels2red=144, out_channels2=288,
                            out_channels3red=32, out_channels3=64, out_channels4=64),
        )

        if self.stage == 'train':
            self.aux_logits2 = InceptionAux(in_channels=528, out_channels=num_class)

        self.block4_3 = nn.Sequential(
            InceptionModule(in_channels=528, out_channels1=256, out_channels2red=160, out_channels2=320,
                            out_channels3red=32, out_channels3=128, out_channels4=128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.block5 = nn.Sequential(
            InceptionModule(in_channels=832, out_channels1=256, out_channels2red=160, out_channels2=320,
                            out_channels3red=32, out_channels3=128, out_channels4=128),
            InceptionModule(in_channels=832, out_channels1=384, out_channels2red=192, out_channels2=384,
                            out_channels3red=48, out_channels3=128, out_channels4=128),
        )

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.linear = nn.Linear(in_features=1024, out_features=num_class)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        aux1 = x = self.block4_1(x)
        aux2 = x = self.block4_2(x)
        x = self.block4_3(x)
        x = self.block5(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        out = self.linear(x)
        if self.stage == 'train':
            aux1 = self.aux_logits1(aux1)
            aux2 = self.aux_logits2(aux2)
            return aux1, aux2, out
        else:
            return out

if __name__ == '__main__':
    model = GoogLeNet_v2()
    print(model)

    input = torch.randn(1, 3, 224, 224)
    aux1, aux2, out = model(input)
    print(aux1.shape)
    print(aux2.shape)
    print(out.shape)


if __name__ == '__main__':
    input = torch.randn(1, 3, 224, 224)
