import torch
import pandas as pd
import datetime
from sklearn import preprocessing
import numpy as np
from matplotlib import pyplot as plt


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


# 开始使用torch提供的高级api搭建sequential model，训练模型
lr = 0.001
epochs = 1000
input_size = input_features.shape[1]
hidden_size = 128
output_size = 1
batch_size = 16
# 搭建网络
model = torch.nn.Sequential(
    torch.nn.Linear(input_size, hidden_size),
    torch.nn.Sigmoid(),
    torch.nn.Linear(hidden_size, output_size)
)
# 配置目标函数和优化器
loss = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
# 开始训练
losses = []  # 记录每一个epoch的平均损失
for epoch in range(epochs):
    batch_loss = []  # 记录每一个batch的损失，用于求平均和表示该epoch的损失
    # 使用 mini-batch的方式进行训练
    for start in range(0, input_features.shape[0], batch_size):
        # 提取batch数据
        end = start + batch_size if start + batch_size<input_features.shape[0] else input_features.shape[0]
        x_train = torch.tensor(input_features[start:end], dtype=torch.float, requires_grad=True)
        y_train = torch.tensor(y[start:end], dtype=torch.float, requires_grad=True)
        # 前向
        '''
        1、UserWarning: Using a target size (torch.Size([64])) that is different to the input size (torch.Size([64, 1])). This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.
        return F.mse_loss(input, target, reduction=self.reduction)
        问题描述：训练时batch大于1 时，loss就不下降，训练效果很差。而batch =1 时可以正常训练。后发现提示警告，预测的batch维度与真实的batch维度不同，按照提示需要统一维度，可用squeeze将预测维度从（64，1）压缩为（64），操作方法为 predicted = model(x_train).squeeze(-1)。
        问题原因：因为线性回归网络的输出是1维，而在读取target数据时，默认读取为了一维向量，而预测的结果是tensor，是在一维的基础上unsuqeeze了batch维度得到的，而在计算mseloss时候，维度不同时计算loss可能导致错误。
        '''
        predicted = model(x_train).squeeze(-1)
        # 计算目标函数值
        loss_value = loss(predicted, y_train)
        batch_loss.append(loss_value.data.numpy())
        # 反向传播
        loss_value.backward(retain_graph=True)
        # 梯度更新
        optimizer.step()

    # 当所有batch跑完，就是一个epoch结束,打印损失
    print(f'epoch: {epoch+1}, loss:{sum(batch_loss)/len(batch_loss)}')
    losses.append(sum(batch_loss)/len(batch_loss))

# 预测结果展示
x_test = torch.tensor(input_features, dtype=torch.float, requires_grad=True)
y_test = model(x_test).data.numpy()
# print(y_test.shape)  # (348, 1) 需要将其转为-维
# 实际的日期和温度值df表格
ture_df = pd.DataFrame({'date': dates, 'tempreture': y})
predict_df = pd.DataFrame({'date': dates, 'tempreture': y_test.reshape(-1)})

# 绘图对比
plt.plot(ture_df['date'], ture_df['tempreture'], 'b-', label='actual')
plt.plot(predict_df['date'], ture_df['tempreture'], 'ro', label='predicted')
plt.xticks(rotation=60)
plt.legend()
plt.xlabel('date')
plt.ylabel('tempreture')
plt.title('predict tempreture')
plt.show()