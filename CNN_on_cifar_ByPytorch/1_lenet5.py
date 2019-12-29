from torch import nn
import torch
class Flatten(nn.Module):
    """
    自定义将特征图展平类
    """
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1)

# 定义网络结构
class Lenet5(nn.Module):
    '''
    CNN--Lenet5 for cifar10 dataset
    '''
    def __init__(self):
        super(Lenet5, self).__init__()  # 继承父类

        # 卷积单元--conv_unit
        self.conv_unit = nn.Sequential(
            # [b, 3, h, w] -> [b, 6, h, w]
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            # [b, 6, h, 2] -> [b, 16, h, w]
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
        )

        # 展平--flatten
        self.flatten = Flatten()

        # 全连接单元--fc-unit
        self.fc_unit = nn.Sequential(
            # 第一层：120
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        # [b, 3, 32, 32] -> [b, 16, 5, 5]
        x = self.conv_unit(x)
        # print(x.shape)
        # [b, 16, 5, 5] -> [b, 16*5*5]
        x = self.flatten(x)
        # print(x.shape)
        # [b, 16*5*5] -> [b, 10]
        x = self.fc_unit(x)
        # print(x.shape)

        # [2, 10]
        return x
