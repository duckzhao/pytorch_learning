import torch
import numpy as np
from torch import nn

# 构建训练集数据
x_values = [i for i in range(11)]
x_train = np.array(x_values, dtype=np.float32)
print('begin shape convert', x_train.shape)
# x train 为二维 数组形式
x_train = x_train.reshape(-1, 1)
print('after shape convert', x_train.shape)

y_values = [2*x+1 for x in x_values]
y_train = np.array(y_values, dtype=np.float32)
y_train = y_train.reshape(-1, 1)

# 构建模型
# torch 中构建模型的方式和tf使用keras构建模型的方式十分相似，都是新建一个model class，然后继承nn.Model，在init中实例化网络组件，在forward
# /(call)中 构建网络实际搭建方式和 tensor 数据传输流
class LinearRegressionModel(nn.Module):
    # 将模型的输入和输出维度，作为model实例化时的变量输入，有助于网络结构的迁移使用（全连接层用得到？）
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        # 开始实例化网络组件
        self.liner = nn.Linear(in_features=input_dim, out_features=output_dim)

    # 利用init的组件，搭建真实 tensor 流动结构
    def forward(self, x):
        out = self.liner(x)
        return out

# 实例化网络class，得到一个model对象，这一步和keras一致
input_dim = 1
output_dim = 1
model = LinearRegressionModel(input_dim, output_dim)

'''
添加如何配置 torch gpu 的方法，首先判断 torch 的gpu-cuda是否配置成功，如果配置成功了，则将模型和gpu进行绑定
'''
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 定义好训练超参数和损失函数
epochs = 1000
lr = 0.01
# 优化器的参数如下 params，指定要优化的参数，一般使用model.parameters()指定，将实例化的model中的带训练参数都设置进来
optimizer = torch.optim.SGD(params=model.parameters(), lr=lr)
# 损失函数定义,这里采用 均方误差
criterion = nn.MSELoss()

for epoch in range(epochs):
    epoch += 1
    # 将网络输入、输出都转换为tensor
    # inputs = torch.from_numpy(x_train)
    # labels = torch.from_numpy(y_train)
    '''
    其次将inputs 和 labels 数据也送入 device中，仅对上述的做法添加 .to(device) 后缀即可，使用这两步 to(device)就 启动了torch的gpu模式了
    '''
    inputs = torch.from_numpy(x_train).to(device)
    labels = torch.from_numpy(y_train).to(device)

    # 每轮训练前将梯度清零，防止梯度累积
    optimizer.zero_grad()

    # 前向传播
    outputs = model(inputs)

    # 计算损失
    loss = criterion(outputs, labels)

    # 反向传播,计算待训练参数的梯度
    loss.backward()

    # 使用梯度计算结果和学习率更新权重参数
    optimizer.step()

    # 每隔一定epochs打印下当前训练情况
    if epoch % 50 == 0:
        print(f'now in epoch: {epoch}, the loss is {loss.item()}')

'''
注意 gpu模式下，使用模型进行预测时，模型的输入 也必须送到 device中去，否则会报错，所以这里 提前 exit 程序
'''
exit()

# 测试模型预测结果
# 其中 requires_grad_()的主要用途是告诉自动求导开始记录对Tensor的操作，虽然不懂加上就行了
# 使用 .data.numpy() 是因为model预测返回的tensor中含有一个 grad_fn属性，使用.data从中仅取出tensor值，在使用numpy()将纯tensor转为nda格式
# predicted = model(torch.from_numpy(x_train).requires_grad_()).data.numpy()
predicted = model(torch.from_numpy(x_train).requires_grad_())
print(predicted)
print(predicted.data)
print(predicted.data.numpy())


# 模型的保存与读取
# 保存模型，仅保存模型的权重参数到本地
torch.save(model.state_dict(), './model.pkl')
print('model weight save success!')
# 当实例化好model对象后，可以这样加载模型参数
model.load_state_dict(torch.load('./model.pkl'))
print('model weight load success!')