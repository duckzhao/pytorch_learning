import torch
from torch import nn
from torch.nn import functional as F

# 继承 nn.Module ,自定义一个模型，相比较 torch.nn.Sequential 这种方式更加灵活，
# Sequential 实际上也是一个 nn.Module 的对象，所以 他们之间可以互相作为另一个的元素（组件）
class MLP(nn.Module):
    # 有关网络层数等超参数可以放到构造函数中
    def __init__(self):
        super().__init__()
        # 在这里定义所有前向推理时要用到的层（一个参数层，定义一次）
        # 定义的顺序无所谓，只要forward的顺序和模型设计一致即可，定义层的变量名会被save到结果中
        # 不带参数的层如relu可以不定义，直接从F中调用即可
        self.hidden = nn.Linear(10, 20)  # 10个变量

        # 在这里也可以添加 torch.nn.Sequential 的对象，Sequential里面也可以添加实例化的net对象
        # 注意relu要使用nn中的组件，F中的组件只能在推理forward时调用
        self.se = nn.Sequential(nn.Linear(20, 40), nn.ReLU(), nn.Linear(40, 20))

        self.out = nn.Linear(20, 2)  # 2 分类问题

    def forward(self, X):
        # 在这里完成前向推理
        return self.out(self.se(F.relu(self.hidden(X))))

# 有重复的block或者layer单元时，可以使用for循环配合 nn.ModuleList完成多个blocks的init定义，
# ModuleList 或者 Sequential 返回的都可以看作是 list 里面存了所有的层（单元）
# forward时，再用 for循环遍历 实例化的变量即可
'''
class Transformer(nn.Module):
    """Transformer with Self-Attentive Blocks"""
    def __init__(self, num_layers, dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for block in self.blocks:
            x = block(x, mask)
        return x
'''

# 实例化网络，网络在实例化的时候会默认给每个layer初始化符合正态分布的权重（每次都不同）
model = MLP()
X = torch.arange(20.).reshape(-1, 10)
ret = model(X)
print(ret)

# 参数管理：访问模型中各层的参数详细信息（名字，权重，）

'''以 Sequential 模型为例，Sequential 返回的实际上是一个list对象，可以使用[]访问某一层layer
class 实例化的网络，不能这样以list形式取'''
net = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 2))

# 模型信息整体查看，是一个list，里面包含了 三个对象，linear，relu，linear
print(net)    # 返回模型中所有定义的layer的信息，返回顺序和定义顺序一致

# 参数访问，state_dict 方法 以OrderedDict形式返回这一层的 weight 和 bias权重具体数值(linear只有这两个参数)
# relu等没有权重的layer，返回空OrderedDict
print(net[2].state_dict())
# 也可以访问某一层具体的参数，如weight，或者bias，访问的结果包含参数值、梯度值，及其他属性信息
# 可以通过 net[2].state_dict()['weight'].data访问真实的参数值，使用.grad访问梯度值
print('weight param is:', net[2].state_dict()['weight'].data)
# 访问梯度值，因为没有反向传播过，因此梯度为0
print('grad param is:', net[2].state_dict()['weight'].grad)


# 一次性访问网络中所有的参数,Sequential 和 class 形式的都可以使用 state_dict 访问

# 这里返回的param 和 state_dict()['weight'] 取出来的一样
# 一次性取网络中某一层的参数信息
print([(name, param.shape) for name, param in net[0].named_parameters()])
# 一次性取Sequential中所有的参数的名字，返回结果中不包含relu层（因为relu没有参数）
print([(name, param.shape) for name, param in net.named_parameters()])

# 拿到net返回的每一层参数的名字之后可以直接在 net.state_dict() 中通过 层参数名字访问参数了
print(net.state_dict()['0.weight'].data)

# 对于class 实例化的model也可以用这种方式取某层对应的参数
# print([(name, param.shape) for name, param in model.named_parameters()])
# print(model.state_dict()['hidden.weight'])

''' 以下 Sequential 和 class 网络操作一致 '''
# 1. 手动权重初始化
# 使用apply函数，初始化网络所有的全连接层权重，apply类似于map，表示对网络中每一层都应用apply函数
def init_normal(m):
    '''
    :param m: 网络中的某一层larer
    :return: 原地修改，不用返回
    '''
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)
        # 还有如下的init方式
        # nn.init.constant_() 常数 、nn.init.uniform_()
net.apply(init_normal)
print(net[0].state_dict())
# network 不管里面还是外面都是 module，因此也可以仅对其中指定的一层进行apply，权重初始化
net[0].apply(init_normal)

# 2. 直接操作权重
net[0].state_dict()['weight'].data += 1  # 全部 +1
net[0].state_dict()['weight'].data[0, 0] = 10   # 修改权重中的某一个值为1
print(net[0].weight)    # 打印权重，net[0].weight 和 net[0].state_dict()['weight'] 引用是一样的

# 3. 多个层共享权重（即一个层用多次）
# Sequential 实现
shared = nn.Linear(10, 10)
new_net = nn.Sequential(nn.Linear(5, 10), nn.ReLU(), shared, nn.ReLU(), shared, nn.Linear(10, 1))
print(new_net[2] == new_net[4])  # True

# class实现，实际上就是在init中定义一个liner，然后在forward中多次推理就行了