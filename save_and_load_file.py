import torch
from torch import nn
from torch.nn import functional as F

# 保存和加载张量
x = torch.arange(4, dtype=torch.float32)
torch.save(x, './x-file')
x2 = torch.load('./x-file')
print(x2 == x)

# 存储一个张量列表
y = torch.zeros(4)
torch.save([x, y], './x-files')
a = torch.load('./x-files')
print(a[0], a[1])

# 存储一个字典
mydict = {'x': x, 'y': y}
torch.save(mydict, './mydict')
mydict2 = torch.load('./mydict')
print(mydict2)


# 存储并加载模型权重
# 定义一个model
class MLP(nn.Module):
    # 有关网络层数等超参数可以放到构造函数中
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(10, 20)  # 10个变量
        self.se = nn.Sequential(nn.Linear(20, 40), nn.ReLU(), nn.Linear(40, 20))
        self.out = nn.Linear(20, 2)  # 2 分类问题

    def forward(self, X):
        # 在这里完成前向推理
        return self.out(self.se(F.relu(self.hidden(X))))

model1 = MLP()
x = torch.arange(10, dtype=torch.float32)
y1 = model1(x)
# 1. 只保存权重(使用state_dict将模型权重信息转换为dict，然后保存dict即可)，需要配合网络定义恢复原始模型
torch.save(model1.state_dict(), "mlp.params")
# 恢复网络需要先实例化 网络结构
model2 = MLP()
model2.load_state_dict(torch.load("mlp.params"))
y2 = model2(x)
print(y1 == y2)  # 相等，说明确实加载了上一次的参数

# 2. 保存网络+权重
torch.save(model1, "mlp.pth")
# 直接读取模型
model3 = torch.load("mlp.pth")
# print(model3)
y3 = model3(x)
print(y1 == y3)