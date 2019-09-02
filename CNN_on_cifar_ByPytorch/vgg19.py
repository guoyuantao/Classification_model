import torch
from torch import nn

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1)

class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()

        # 卷积单元
        self.conv_unit = nn.Sequential(
            # 第一个卷积单元
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=1),

            # 第二个卷积单元
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=1),

            # 第三个卷积单元
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=1),

            # 第四个卷积单元
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第五个卷积单元
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 展平
        self.flatten = Flatten()

        # 全连接单元
        self.fc_unit = nn.Sequential(
            # 第一个全连接层
            nn.Linear(512*7*7, 4096),
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
        # 展平操作
        x = self.flatten(x)
        # 全连接操作
        x = self.fc_unit(x)

        return x

