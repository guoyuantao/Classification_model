import torch
from torch import nn

class Inception(nn.Module):
    def __init__(self, in_ch, branch_ch_1, branch_ch_2_1, branch_ch_2_2, branch_ch_3_1, branch_ch_3_2, branch_ch_4):
        super(Inception, self).__init__()

        # Inception结构
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, branch_ch_1, kernel_size=1)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, branch_ch_2_1, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(branch_ch_2_1, branch_ch_2_2, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_ch, branch_ch_3_1, kernel_size=1, stride=1, padding=1),
            nn.Conv2d(branch_ch_3_1, branch_ch_3_2, kernel_size=5, stride=1, padding=1),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Conv2d(in_ch, branch_ch_4, kernel_size=1, padding=1),
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)

        # 输出连接
        x = torch.cat([x1, x2, x3, x4], 1)

        return x

class GoogLeNet_v1(nn.Module):
    def __init__(self):
        super(GoogLeNet_v1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
        )
        self.inception_3 = nn.Sequential(
            Inception(192, 64, 96, 128, 16, 32, 32),
            Inception(256, 128, 128, 192, 32, 96, 64),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
        )
        self.inception_4 = nn.Sequential(
            Inception(480, 192, 96, 208, 16, 48, 64),
            Inception(512, 160, 112, 224, 24, 64, 64),
            Inception(512, 128, 128, 256, 24, 64, 64),
            Inception(512, 112, 144, 288, 32, 64, 64),
            Inception(528, 256, 160, 320, 32, 128, 128),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
        )
        self.inception_5 = nn.Sequential(
            Inception(832, 256, 160, 320, 32, 128, 128),
            Inception(832, 384, 192, 384, 48, 128, 128),
        )
        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.droput = nn.Dropout(p=0.4)
        self.linear = nn.Linear(1024*5*5, 10)

    def forward(self, x):
        # 卷积操作
        x = self.conv1(x)
        # 卷积操作
        x = self.conv2(x)
        # 第一个inception块
        x = self.inception_3(x)
        # 第二个inception块
        x = self.inception_4(x)
        # 第三个inception块
        x = self.inception_5(x)
        # 平均池化
        x = self.avg_pool(x)
        # 连续两个缩减维度
        x = torch.squeeze(x, dim=-1)
        x = torch.squeeze(x, dim=-1)
        # 随机失活
        x = self.droput(x)
        x = x.view(x.size(0), -1)
        # 线性层
        x = self.linear(x)

        return x

