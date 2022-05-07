import torch
from torch import nn
from torch.nn import functional as F

# 构建一个层实际上和构建一个网络是一样的，因为都是nn.Module子类，且都可以相互嵌套
class CenterLayer(nn.Module):
    # 构造函数中可以有参数，也可以没参数
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()

# 实例化自定义层，和实例化class model是一样的
layer = CenterLayer()
print(layer(torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)))

net = nn.Sequential(nn.Linear(8, 128), CenterLayer())
X = torch.rand(4, 8)
# print(X.shape)    4, 8
Y = net(X)
print(Y.mean())


# 使用torch基础的 参数数组初始化api及矩阵乘法api自定义layer
class MyLinear(nn.Module):
    def __init__(self, in_feature: int, out_feature: int):
        super().__init__()
        # 如果要自定义参数，而不是用nn.layer等已经写好的api，则注意要用 nn.Parameter 修饰
        # 使用 nn.Parameter 修饰参数矩阵，会给他们默认加上 requires_grad，以及参数名的后期管理等等
        # 初始化W矩阵
        self.weights = nn.Parameter(torch.randn(in_feature, out_feature))
        # 初始化bais
        self.bias = nn.Parameter(torch.randn(out_feature))

    def forward(self, X):
        # 注意对 nn.Parameter 对象参数进行运算时，必须使用.data访问到实际的tensor参与运算，nn.Parameter 对象不能直接参与矩阵运算
        y = X @ self.weights.data + self.bias.data
        # 或者 y = torch.matmul(X, self.weights.data) + self.bias.data
        return F.relu(y)

# 自定义层的正向传播
myl = MyLinear(3, 3)
X = torch.arange(3, dtype=torch.float32).reshape(1, 3)
print(myl(X))
