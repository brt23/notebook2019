import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import sys
import time


"""
本文实现了进度条功能
"""

# 构建dataloader
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

trainset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=4,
    shuffle=True,
    num_workers=2
)

testset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=4,
    shuffle=False,
    num_workers=2
)

classes = (
    'plane', 'car', 'bird', 'cat',
    'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
)


def imshow(image):
    image = image / 2 + 0.5
    np_image = image.numpy()
    plt.imshow(np.transpose(np_image, (1, 2, 0)))
    plt.show()

dataiter = iter(trainloader)
images, labels = dataiter.next()
print(images.size())
imshow(torchvision.utils.make_grid(images))
print(" ".join('%5s' % classes[labels[j]] for j in range(4)))

# 构建网络
class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=3,
            out_channels=6,
            kernel_size=5
        )
        self.pool = torch.nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5
        )
        self.fc1 = torch.nn.Linear(
            in_features=16 * 5 *5,
            out_features=120
        )
        self.fc2 = torch.nn.Linear(
            in_features=120,
            out_features=84
        )
        self.fc3 = torch.nn.Linear(
            in_features=84,
            out_features=10
        )

    # 前向传播
    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 实例化网络
net = Net()
# 设置损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 设置进度条相关设置参数和超参数
epochs = 2
data_len = len(trainloader)
bar_width = 50                  # 进度条在终端中的长度
dis_loss = None                 # 进度条显示的loss值
dis_per_time = 1                # 进度条loss每1秒刷新一次
last_time = time.process_time() # 用于计算显示刷新的前时间点

for epoch in range(epochs):
    print('epochs: {}/{}'.format(epoch+1, epochs))
    running_loss = 0.0
    running_loss_count = 0
    epoch_loss = 0.0
    epoch_loss_count = 0

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward() 
        optimizer.step()

        running_loss += loss.item()
        running_loss_count += 1
        epoch_loss += loss.item()
        epoch_loss_count += 1

        now_time = time.process_time()                  # 用于计算显示刷新的后时间点
        progress = int(bar_width * (i+1) / data_len)    # 计算的进度条进度
        if dis_loss is None:
            dis_loss = loss.item()
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # 进度条
            sys.stdout.write('{:2}{}{:5}/{:5}: '.format(' ', 'steps: ', i+1, data_len))
            sys.stdout.write('[{}{}]'.format('#' * progress, " " * (bar_width - progress)))
            sys.stdout.write(' - loss: {:.3f}'.format(dis_loss))
            sys.stdout.write('\r')
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        elif progress == bar_width:
            dis_loss = running_loss / running_loss_count # 求出此时间区间的平均loss
            dis_epoch_loss = epoch_loss / epoch_loss_count
            running_loss = 0.0
            running_loss_count = 0
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # 进度条
            sys.stdout.write('{:2}{}{:5}/{:5}: '.format(' ', 'steps: ', i+1, data_len))
            sys.stdout.write('[{}{}]'.format('#' * progress, " " * (bar_width - progress)))
            sys.stdout.write(' - loss: {:.3f} - epoch loss: {:.3f}'.format(dis_loss, dis_epoch_loss))
            sys.stdout.write('\r')
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            sys.stdout.write('\n')                 # 当结束时，换一行以保存数据，以免最后的数据被刷新掉
        elif now_time - last_time >= dis_per_time: # 如果时间差大于等于刷新间隔
            last_time = time.process_time()        # 跟新前时间点
            dis_loss = running_loss / running_loss_count # 求出此时间区间的平均loss
            running_loss = 0.0
            running_loss_count = 0
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # 进度条
            sys.stdout.write('{:2}{}{:5}/{:5}: '.format(' ', 'steps: ', i+1, data_len))
            sys.stdout.write('[{}{}]'.format('#' * progress, " " * (bar_width - progress)))
            sys.stdout.write(' - loss: {:.3f}'.format(dis_loss))
            sys.stdout.write('\r')
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            
        sys.stdout.flush() # 刷新标准输出缓存，以达到在原地跟新进度的状态

print('Finished Training')


test_dataiter = iter(testloader)
test_images, test_labels = test_dataiter.next()
imshow(torchvision.utils.make_grid(test_images))
print('GroundTruth: ', ' '.join('%5s' % classes[test_labels[j]] for j in range(4)))

test_outputs = net(test_images)
_, test_predicted = torch.max(test_outputs, 1)
print('Predicted: ', ' '.join('%5s' % classes[test_predicted[j]] for j in range(4)))