import torch


# scalar 标量，又称数值
scalar = torch.tensor(37, dtype=torch.float32)
print(scalar, '他的维度是：', scalar.dim())
print('scalar 的基本运算：', scalar*2)
print('取出他的数值类型结果:', scalar.item())

# vector 向量
vector = torch.tensor([1, 2, 3])
# dim() 取出维度，这个向量是一维的
print(vector.dim())
print(vector.size())

# matrix 矩阵 多维的
matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(matrix)
# 矩阵乘法 matrix.T表示矩阵的转置
print(matrix.matmul(matrix.T))
# 元素乘法
print(matrix*matrix)