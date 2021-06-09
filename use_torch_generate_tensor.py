import torch
import numpy as np

# 产生一个指定大小的空 tensor，但是系统会默认给里面填充一些接近0的小数
# 注意这里输入 size时，是依次输入的 每个维度的大小，而不是用tuple打包size输入
x = torch.empty(5, 3)
print(x)
# tensor.size() 等同于其他框架里面的 tensor.shape 返回当前tensor 的维度大小
print(x.size(), x.dtype)


# 产生指定大小的 随机数 张量,包含了从区间[0, 1)的均匀分布中抽取的一组随机数。张量的形状由参数sizes定义。
y = torch.rand(5, 3)
print(y)

# 返回一个张量，包含了从标准正态分布（均值为0，方差为1，即高斯白噪声）中抽取的一组随机数。张量的形状由参数sizes定义
y2 = torch.randn(5, 3)
print(y2)


# 创建指定大小的 全0 矩阵,torch.long 实际上就是 torch.int64
z = torch.zeros(5, 3, dtype=torch.long)
print(z, z.dtype)


# 直接传入 现有矩阵 ，转为tensor
a = torch.tensor(data=[[1, 3, 5]], dtype=torch.float32)
print(a, a.size())


# tensor.new_ones（size） , 对一个已有的tensor执行该函数，返回一个 与 size 相同大小的 全1 张量，其意义在于？不懂，默认新张量和tensor同dtype
b = a.new_ones(3, 3)
print(b)
print(a)    # 该方法不会覆盖原始张量 a，不对 a 产生任何影响


# tensor 的 索引 ，和 numpy 基本一致
print(y)
print(y[0, :])  # 索引第 0 行的所有列


# view改变tensor的维度，不会对原始tensor产生影响，在tf中是使用 tf.reshape 完成 shape的改变，注意改变前后矩阵中元素数量不变
x = torch.randn(4, 4)
y = x.view(16)  # 拉直成一维
z = x.view(-1, 8)   # 保证第二维是 8，然后自动计算第一维的值
print(x.size(), y.size(), z.size())

# 与 numpy 协同操作，tensor和numpy类型转换
a = torch.ones(4, 4)
b = a.numpy()
print(type(a), type(b))

c = np.arange(12).reshape(3, 4)
print(c)
d = torch.from_numpy(c)
print(type(c), type(d))

# tensor 四则运算，这里和 tf 也基本保持一致
x = torch.randn(4, 4)
y = torch.rand(4, 4)
print(x+y, x-y)