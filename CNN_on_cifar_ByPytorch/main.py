import torch

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch import nn, optim

from lenet5 import Lenet5
from alexnet import Alexnet
from vgg16 import VGG16
from vgg19 import VGG19
from resnet18 import ResNet18
from goolenet_v1 import GoogLeNet_v1
def main():
    '''
    主函数
    :return:
    '''
    # 参数设置
    batch_size = 128

    # 加载数据
    # 训练数据
    cifar_train = datasets.CIFAR10('cifar', True, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]), download=True)
    cifar_train = DataLoader(cifar_train, batch_size=batch_size, shuffle=True)

    # 测试数据
    cifar_test = datasets.CIFAR10('cifar', False, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]), download=True)
    cifar_test = DataLoader(cifar_test, batch_size=batch_size, shuffle=True)

    # 查看数据
    data, label = iter(cifar_train).next()
    print("data shape: ", data.shape, 'label shape: ', label.shape)
    print("Train Length: ", len(cifar_train.dataset), "Test Length: ", len(cifar_test.dataset))

    # 将模型转移到GPU上
    device = torch.device('cuda')

    # ============== 此处更换模型 ==================
    # model = Lenet5().to(device)
    # model = Alexnet().to(device)
    # model = VGG16().to(device)
    # model = VGG19().to(device)
    # model = ResNet18().to(device)
    model = GoogLeNet_v1().to(device)
    # ============================================

    # 优化器与损失值
    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print(model)

    for epoch in range(1000):

        # 训练
        model.train()
        for batch_idx, (data, label) in enumerate(cifar_train):
            # 将训练数据，训练标签转移到GPU
            data, label = data.to(device), label.to(device)

            # 计算模型输出值,[b, 10]
            logits = model(data)

            # 计算损失值 logits:[b, 10], label:[b, 10]
            loss = criteon(logits, label)

            # 计算梯度
            optimizer.zero_grad()    # 梯度清零
            loss.backward()          # 计算梯度
            optimizer.step()         # 梯度更新

            # 输出训练信息
            if batch_idx % 100 == 0:
                print("Train epoch: {} [{}/{} ({:.0f}%)] \t Loss: {:.6f}"
                      .format(epoch, batch_idx*batch_size, len(cifar_train.dataset),
                              100. * batch_idx / len(cifar_train), loss.item()))

        # 测试
        model.eval()
        with torch.no_grad():
            total_correct = 0   # 预测正确总数
            total_num = 0       # 数据总数
            test_loss = 0       # 测试平均损失
            for data, label in cifar_test:
                # 将测试数据转移到GPU
                data, label = data.to(device), label.to(device)

                # 计算预测值 [b, 10]
                logits = model(data)

                # 计算损失值
                test_loss += criteon(logits, label).item()

                # 求得最大值对应的下标
                predict = logits.argmax(dim=1)

                # 求得预测正确的总数
                correct = torch.eq(predict, label).float().sum().item()
                total_correct += correct
                total_num += data.size(0)

            test_loss /= len(cifar_test.dataset)
            acc = total_correct / total_num

            print('Test set: Average loss: {}, Accuracy: {}%'
                  .format(test_loss, 100. * acc))




if __name__ == '__main__':
    main()