import torch
import pandas as pd
import datetime
from matplotlib import pyplot as plt
from sklearn import preprocessing
import numpy as np

# 读取数据
ori_df = pd.read_csv('./temps.csv')
# print(ori_df.head(5))
# year  month  day  week  temp_2  temp_1  average  actual  friend
# print(ori_df.shape) # (348, 9)

# 读取df中的时间标签，并合并转换为新的时间格式
years = ori_df['year']
months = ori_df['month']
days = ori_df['day']
# 转为datetime格式
dates = [str(_[0])+'-'+str(_[1])+'-'+str(_[2]) for _ in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]
# print(dates[:5])

# 绘置实际天气温度 和 相关变量 之间的趋势 对比图
# 设置默认绘图风格
plt.style.use('fivethirtyeight')
plt.figure(figsize=(10, 10))
# actual
# 绘图时可以传入dates格式的变量作为x
plt.subplot(2, 2, 1)
plt.plot(dates, ori_df['actual'])
plt.ylabel('tempreture')
plt.title('actual')
# 传入空列表以取消xticks
plt.xticks([])
# yesterday
plt.subplot(2, 2, 2)
plt.plot(dates, ori_df['temp_1'])
plt.ylabel('tempreture')
plt.title('yesterday')
plt.xticks([])
# twodaysago
plt.subplot(2, 2, 3)
plt.plot(dates, ori_df['temp_2'])
plt.ylabel('tempreture')
plt.xlabel('date')
plt.title('twodaysago')
plt.xticks(rotation=30)
# friend
plt.subplot(2, 2, 4)
plt.plot(dates, ori_df['friend'])
plt.ylabel('tempreture')
plt.xlabel('date')
plt.title('friend')
plt.xticks(rotation=30)

# 子图间隔
plt.tight_layout(pad=2)

plt.show()


# 数据预处理
# 将 英文属性值的类别 转为one-hot编码，以便输入model
features = pd.get_dummies(ori_df)
# print(features.head(5))

# 将x和y分割开来
y = np.array(features['actual'])
x = features.drop(labels='actual', axis=1)
# print(y, x)

# 归一化处理，减少属性之间 因取值数量级 所带来的收敛误差
# sklearn也仅接收ndarray格式的输入，所以转df为ndarray
features = np.array(x)
input_features = preprocessing.StandardScaler().fit_transform(features)
# print(input_features)


# 开始不借助 torch.nn api手动搭建简单的 全连接层model

# 将 model 训练的x y 都转为 tensor格式
x = torch.tensor(input_features)
y = torch.tensor(y)

# 将需要训练的 权重参数都进行初始化
# 14 个属性，连接到 128个 units 上，所以权重矩阵为 14*128,每个神经元一个偏置共128个,该参数需要反向传播训练，所以requires_grad=True
weights1 = torch.randn(14, 128, dtype=torch.double, requires_grad=True)
bias1 = torch.randn(128, dtype=torch.double, requires_grad=True)
# 由于是回归问题，再将128个神经元连接到一个units的输出上进行输出回归
weights2 = torch.randn(128, 1, dtype=torch.double, requires_grad=True)
bias2 = torch.randn(1, dtype=torch.double, requires_grad=True)

# 反向传播过程设计
epochs = 1000
lr = 0.001
losses = [] # 训练epochs损失记录列表
for epoch in range(epochs):
    # 前向传播计算
    hidden = x.mm(weights1) + bias1
    # 激活
    hidden = torch.relu(hidden)
    # 输出
    predict = hidden.mm(weights2) + bias2
    # 计算损失，采用mse目标函数
    loss = torch.mean((y-predict)**2)
    losses.append(loss.data.numpy())
    # 打印每一轮的损失值
    print(f'epoch: {epoch+1}, loss：{loss.data}')

    # 反向传播，更新参数
    loss.backward()
    # 请注意，这里是add_，而不是add！！！否则loss值不变，weights没有被优化
    weights1.data.add_(- lr*weights1.grad.data)
    weights2.data.add_(- lr*weights2.grad.data)
    bias1.data.add_(- lr*bias1.grad.data)
    bias2.data.add_(- lr*bias2.grad.data)

    # 清空梯度
    weights1.grad.data.zero_()
    weights2.grad.data.zero_()
    bias1.grad.data.zero_()
    bias2.grad.data.zero_()
