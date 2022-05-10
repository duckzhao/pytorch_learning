import torch
from torch import nn
from torch.nn import functional as F

''''1*1， 3*3，5*5卷积核用哪个？小孩子才做选择，我全都要
1. 在inception block中使用4条不同大小卷积核的路径提取不同的特征，并进行汇总
2. 使用大量 1*1 的卷积降低卷积层输入的channel，减少参数量，使用了GAP
3. 第一个达到100层+的 model（算上横向的），并且后续有一系列包含新算法的改进，与时俱进

p.s. 卷积层超参数设计很怪异，估计是搜索搜出来的
'''

class inception_block(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4):
        super(inception_block, self).__init__()
        self.p1 = nn.Conv2d(in_channels, c1, 1, 1, 0)
        self.p2_1 = nn.Conv2d(in_channels, c2[0], 1, 1, 0)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], 3, 1, 1)
        self.p3_1 = nn.Conv2d(in_channels, c3[0], 1, 1, 0)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], 5, 1, 2)
        self.p4_1 = nn.MaxPool2d(3, 1, 1)
        self.p4_2 = nn.Conv2d(in_channels, c4, 1, 1, 0)

    def forward(self, X):
        p1 = F.relu(self.p1(X))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(X))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(X))))
        p4 = F.relu(self.p4_2(self.p4_1(X)))    # 池化后面不用接relu
        return torch.cat((p1, p2, p3, p4), dim=1)   # (b, c, h, w) channel维度拼接是1

# (192, 64, (96, 128), (16, 32), 32)
# inc_blk = inception_block(*(192, 64, (96, 128), (16, 32), 32))
# print(inc_blk)

b1 = nn.Sequential(nn.Conv2d(3, 64, 7, 2, 3),
                   nn.ReLU(),
                   nn.MaxPool2d(3, 2, 1))
b2 = nn.Sequential(nn.Conv2d(64, 64, 1, 1, 0),
                   nn.ReLU(),
                   nn.Conv2d(64, 192, 3, 1, 1),
                   nn.ReLU(),
                   nn.MaxPool2d(3, 2, 1))
b3 = nn.Sequential(inception_block(192, 64, (96, 128), (16, 32), 32),
                   inception_block(256, 128, (128, 192), (32, 96), 64),
                   nn.MaxPool2d(3, 2, 1))
b4 = nn.Sequential(inception_block(480, 192, (96, 208), (16, 48), 64),
                   inception_block(512, 160, (112, 224), (24, 64), 64),
                   inception_block(512, 128, (128, 256), (24, 64), 64),
                   inception_block(512, 112, (144, 288), (32, 64), 64),
                   inception_block(528, 256, (160, 320), (32, 128), 128),
                   nn.MaxPool2d(3, 2, 1))
b5 = nn.Sequential(inception_block(832, 256, (160, 320), (32, 128), 128),
                   inception_block(832, 384, (192, 384), (48, 128), 128),
                   nn.AdaptiveAvgPool2d(1),
                   nn.Flatten())
model = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))

# 该模型使用GAP压缩了feature map的W,H，在channel上连接了全连接层，因此不关注输入图片的大小都能跑起来
X = torch.randn((1, 3, 224, 224))

for layer in model:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)