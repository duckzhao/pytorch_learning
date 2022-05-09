import torch
from torch import nn
from torch.nn import functional as F
import torchvision

'''对后续网络设计的思想启发：
1. 使用可重复的块（封装为一个sequential或者class module），来构建深度学习网络
2. 设置很多不同的配置用于构建网络，高配版（刷榜）、低配版（速度快）
'''

def vgg_block(num_convs, in_channels, out_channels):
    layers = list()
    for i in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1))
        layers.append(nn.ReLU())  # 因为最后要放到sequential中，因此使用nn.RELU
        in_channels = out_channels  # 保证下一层的inchannel 和 上一层 out_channels匹配
    layers.append(nn.MaxPool2d(2, 2))
    # 将list的layers中的层都放到Sequential结构中，否则计算图中的param无法被pytorch记录
    return nn.Sequential(*layers)


def vgg(conv_arch):
    conv_blocks = list()
    in_channels = 3  # 如果是灰度图应该改为1
    # 将卷积层按照架构堆叠
    for (num_convs, out_channels) in conv_arch:
        conv_blocks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels  # 本层的out即下层的in
    # 使用sequential结构解析list中的层，并堆叠其余固定的全连接层
    return nn.Sequential(*conv_blocks, nn.Flatten(), nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
                       nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5), nn.Linear(4096, 10))


if __name__ == '__main__':
    conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
    model = vgg(conv_arch).cuda()

    # 跑一次前向看看具体的结果
    X = torch.randn((1, 3, 224, 224)).cuda()
    # 只有sequential结构才可以这样迭代(class module使用named_children)，使用每一层layer单独运算，并拿到结果
    for blk in model:
        X = blk(X)
        print(blk.__class__.__name__, 'output shape:\t', X.shape)
    print(X.device)

    print('**********************class module**********************')

    X = torch.randn((1, 3, 224, 224)).cuda()
    model = torchvision.models.vgg16(pretrained=True).cuda()
    for name, blk in model.named_children():
        X = blk(X)
        # 补齐缺少的flatten层
        if name == 'avgpool':
            X = X.reshape(-1, 25088)
        print(name, 'output shape:\t', X.shape)