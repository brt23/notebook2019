import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


"""
使用GPU训练网络
使用torch的device接口
"""

# 超参数
epochs_ = 1
batch_size_ = 64


trainsets = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

trainloader = torch.utils.data.DataLoader(
    trainsets,
    batch_size=batch_size_,
    shuffle=True,
    num_workers=2
)

testsets = torchvision.datasets.MNIST(
    root = './data',
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

testloader = torch.utils.data.DataLoader(
    testsets,
    batch_size=batch_size_,
    shuffle=False,
    num_workers=2
)



class MNIST(nn.Module):

    def __init__(self):
        super(MNIST, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=5
        )

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=5
        )

        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=5
        )

        self.dense1 = nn.Linear(
            in_features=32 * 16 * 16,
            out_features=256
        )

        self.dense2 = nn.Linear(
            in_features=256,
            out_features=64
        )

        self.dense3 = nn.Linear(
            in_features=64,
            out_features=10
        )

        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = x.view(-1, 32 * 16 * 16)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.relu(x)
        x = self.dense3(x)
        return x


net = MNIST()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)

# 设置设备
# 如果cuda可用，使用cuda，否则使用cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 把网络模块和参数转换为cuda张量
net.to(device)


for epoch in range(epochs_):

    running_loss = 0.0
    running_loss_count = 0
    for step, data in enumerate(trainloader):
        inputs, labels = data
        # 同时输入的训练数据和标签也需要转换为cuda张量
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad() # 清空上一步的残余更新参数值，应为在pytorch中变量是积累的，所以要手动归零
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()  # 反向传播求导
        optimizer.step() # 跟新权重
        
        running_loss += loss.item()
        running_loss_count += 1

        if step % 100 == 99:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, step + 1, running_loss / running_loss_count))
            running_loss = 0.0


# 在测试时可以把网络模块和参数转换为cpu张量
net.to('cpu')
test_dataiter = iter(testloader)
test_images, test_labels = next(test_dataiter)

test_outputs = net(test_images)
# 注意max函数返回一个元组，元组第一个元素是最大值向量，元组第二个元素是最大元素下标向量
_, test_predict = torch.max(test_outputs, 1) 
print(test_predict)
print()
print(test_labels)