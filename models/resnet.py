from torch import nn
import torch
from torch.nn import functional as F

'''保证新加入的层学到的东西不会更差
残差思想将top的梯度能够传播回去深层layer，更新深层layer参数，使得能够训练更深的网络；
随后的所有网络设计都借鉴了这一思想，cnn、mlp、transformer。
'''

class residual_block(nn.Module):
    # 通道数翻倍，宽高减半一般是在第一个卷积中做的
    def __init__(self, input_channels, num_channels, use_1x1_conv=False, strides=1):
        super(residual_block, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, 3, strides, 1)
        self.bn1 = nn.BatchNorm2d(num_channels)  # 参数是输入特征图的 channel数，即第二维
        self.conv2 = nn.Conv2d(num_channels, num_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(num_channels)
        if use_1x1_conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, 1, strides, 0)
        else:
            self.conv3 = None

    def forward(self, X):
         Y1 = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(X)))))
         if self.conv3:
             Y2 = self.conv3(X)
         else:
             Y2 = X
         return F.relu(Y1+Y2)

# 不改变 输出特征图的shape和channel
block = residual_block(3, 3).cuda()
X = torch.randn((4, 3, 224, 224)).cuda()
print(block(X).shape)

# 改变特征图的shape和channel
block2 = residual_block(3, 6, True, 2).cuda()
print(block2(X).shape)