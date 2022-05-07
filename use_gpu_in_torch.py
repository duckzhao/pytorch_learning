import torch
from torch import nn
from torch.nn import functional as F

# 查看当前torch版本信息
print(torch.__version__)

# 查看gpu是否可用
print(torch.cuda.is_available())

# 查看共有几块gpu
print(torch.cuda.device_count())

# 没有指定gpu时，深度学习框架都默认用cpu
# device = torch.device('cpu')
# 返回一个gpu设备
device = torch.device('cuda:0')
# print(device)

# 不指定device时，默认是在cpu上生成的tensor
a = torch.arange(12).reshape((3, 4))
print(type(a.device), a.device)

# 指定device创建张量
b = torch.arange(12, device=device, dtype=torch.float32).reshape((3, 4))
print(type(b.device), b.device)

# 两个张量进行运算时，必须保证他们处于相同的device上，此时a在cpu，b在cuda:0上，故无法完成运算，报错
# c = a + b
# 将a转到cuda变量，a1，然后使用a1和b完成运算即可，算完的结果c也会在cuda:0上
a1 = a.cuda(0)
c = a1 + b
# print(c)    # gpu数据也可以print打印出来
# 使用 .cpu将gpu上的数据拷贝回cpu上
c1 = c.cpu()
print(type(c1.device), c1.device)

# 如何将网络定义在gpu上
net = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1))
# 将网络移到某个device
net = net.to(device)
# 此时网络的所有weight也在gpu上
print(net[0].weight.data)


# 前向推理(传入的x必须和网络处于同一设备)
print(net(torch.tensor([1, 2], dtype=torch.float32, device=device)))

'''
数据的预处理最好在cpu上操作，因为有些操作gpu不一定支持，一般是在把数据送给网络之前才移到gpu上的
'''