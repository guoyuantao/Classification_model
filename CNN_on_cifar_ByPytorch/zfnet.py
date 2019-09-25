import torch
from torch import nn

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1)

class ZFNet(nn.Module):
    def __init__(self):
        super(ZFNet, self).__init__()

        # 卷积单元
        self.conv_unit = nn.Sequential(
            # 第一层卷积
            nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),

            # 第二层卷积
            nn.Conv2d(96, 256, kernel_size=5, stride=2, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # 第三层卷积
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),

            # 第四层卷积
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),

            # 第五层卷积
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        # 展平操作
        self.flatten = Flatten()

        # 全连接层单元，三个全连接层
        self.fc_uint = nn.Sequential(
            # 第一层全连接层
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.6),

            # 第二个全连接层
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.6),

            # 第三个全连接层
            nn.Linear(4096, 10),
        )


    def forward(self, x):
        # 卷积操作
        x = self.conv_unit(x)
        # print(x.shape)
        # 展平操作
        x = self.flatten(x)
        # print(x.shape)
        # 全连接操作
        x = self.fc_uint(x)
        # print(x.shape)

        return x
