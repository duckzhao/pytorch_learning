import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from d2l import torch as d2l
from torch.utils import data
import torchvision

'''
如果只简单的使用 nn.DataParallel，虽然很简单就可以加速，但实际上gpu并行度并不高，是一个gpu算完，才去算另一个gpu，单进程多线程
官方也是推荐使用 nn.DistributedDataParallel 这种方式进行gpu并行，使用的是 多进程，避免了GIL
https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html?highlight=dataparallel#torch.nn.DataParallel
'''

def get_all_gpus():
    gpu_num = torch.cuda.device_count()
    if gpu_num >= 1:
       return [torch.device('cuda:' + str(i)) for i in range(gpu_num)]
    return torch.device('cpu')
# print(get_all_gpus())

def load_data_fashion_mnist(batch_size, resize=None):
    """Download the Fashion-MNIST dataset and then load it into memory.

    Defined in :numref:`sec_fashion_mnist`"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="~/download/", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="~/download/", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=4),
            data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=4))

# 定义模型
def resnet18(num_classes, in_channels=1):
    """稍加修改的ResNet-18模型"""
    def resnet_block(in_channels, out_channels, num_residuals,
                     first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(d2l.Residual(in_channels, out_channels,
                                        use_1x1conv=True, strides=2))
            else:
                blk.append(d2l.Residual(out_channels, out_channels))
        return nn.Sequential(*blk)

    # 该模型使用了更小的卷积核、步长和填充，而且删除了最大汇聚层
    net = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU())
    net.add_module("resnet_block1", resnet_block(
        64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))
    net.add_module("resnet_block4", resnet_block(256, 512, 2))
    net.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1,1)))
    net.add_module("fc", nn.Sequential(nn.Flatten(),
                                       nn.Linear(512, num_classes)))
    return net
model = resnet18(10)


def train(net, batch_size, lr):
    # fashion mnist图片太小了，需要放大才可以被vgg跑
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    devices = get_all_gpus()
    def init_weights(m):
        if type(m) in [nn.Linear, nn.Conv2d]:
            nn.init.normal_(m.weight, std=0.01)
    net.apply(init_weights)

    # mutigpu: 1. 在多个GPU上设置模型
    net = net.to(devices[0])    # 首先需要将模型转到gpu上，否则会报错
    net = nn.DataParallel(net, device_ids=devices)

    trainer = torch.optim.SGD(net.parameters(), lr)
    loss = nn.CrossEntropyLoss()
    timer, num_epochs = d2l.Timer(), 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    for epoch in range(num_epochs):
        net.train()
        timer.start()
        for X, y in train_iter:
            trainer.zero_grad()
            # mutigpu: 2. 虽然这里是将 训练集转到cuda0上，但实际torch会将X在 送入网络 的时候均匀分配到多张卡上（这句话必须有，否则报错）
            X, y = X.to(devices[0]), y.to(devices[0])

            l = loss(net(X), y)
            l.backward()
            trainer.step()
        timer.stop()
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(net, test_iter),))
    print(f'测试精度：{animator.Y[0][-1]:.2f}，{timer.avg():.1f}秒/轮，'
          f'在{str(devices)}')

train(model, batch_size=512, lr=0.01)