import torch
from torch import nn
from torch.nn import functional as F

'''普通卷积核：窗口特征提取的更好、参数共享降低参数量
1*1卷积的兴起，卷积可以看作一种特殊的全连接，大全连接层参数量大，会较容易拟合网络，导致过拟合；
在大卷积的输出特征图上进行使用两个1*1卷积，增强了大卷积输出的每个像素通道的非线性表达力（每个1*1后都有relu）；
剔除全连接层，使用1*1卷积+全局平均池化GAP来代替，不容易过拟合，且极大程度降低了参数量.

虽然它的精度不高，但是他的思想在后面影响很大！！！
'''

class nin_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernal_size, strides, padding):
        super().__init__()
        self.con2d_1 = nn.Conv2d(in_channels, out_channels, kernal_size, strides, padding)
        # 两个1*1卷积不改变channel数
        self.con2d_2 = nn.Conv2d(out_channels, out_channels, 1)
        self.con2d_3 = nn.Conv2d(out_channels, out_channels, 1)

    def forward(self, X):
        return F.relu(self.con2d_3(F.relu(self.con2d_2(F.relu(self.con2d_1(X))))))

# 这个基础的网络结构来自alexnet
model = nn.Sequential(nin_block(3, 96, kernal_size=11, strides=4, padding=0), nn.MaxPool2d(3, stride=2),
                      nin_block(96, 256, 5, 1, 2), nn.MaxPool2d(3, stride=2),
                      nin_block(256, 384, 3, 1, 1), nn.MaxPool2d(3, stride=2),
                      nn.Dropout(0.5),
                      nin_block(384, 10, 3, 1, 1),
                      nn.AdaptiveAvgPool2d(output_size=1),
                      nn.Flatten())
model = model.cuda()
# print(model)
X = torch.randn((1, 3, 224, 224)).cuda()
for name, blk in model.named_children():
    X = blk(X)
    print(name, blk.__class__.__name__, 'output shape:\t', X.shape)