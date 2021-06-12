import torch

# 方法一：对于需要求导（优化）的 计算变量，可以手动声明 “可求导”
# requires_grad=True 参数用于指定 x 变量可以求导---torch会自动解析 x 为可求导变量，而非constant
x = torch.randn(3, 4, requires_grad=True)
# 对于可求导变量，打印时会有 requires_grad=True 的标识
print(x)

# 方法二，在变量声明后，修改其requires_grad属性，将x声明为可求导变量
x = torch.randn(3, 4)
x.requires_grad = True
print(x)

# 一个自动求导机制的求导案例
x = torch.randn(3, 4, requires_grad=True)
b = torch.randn(3, 4, requires_grad=True)
t = x + b
print(t, t.requires_grad)
y = t.sum()
print(y)

# 使用 tensor.backward() 对tensor中 requires_grad 标记为True的张量进行求导运算，在计算的最后一层调用即可对前面的变量都自动求导
# 然后可以使用b.grad 属性 来获取 与该 tensor 相关的变量的倒数值
y.backward()
# x 和 b 会被自动计算倒数，使用如下方式可以获得其倒数值,但是虽然t的requires_grad是true，但是他的导数值却是None，可能是y和t之间不存在变量关系
print(x.grad, b.grad, t.grad)
# 查看上述变量的 requires_grad 属性值，由 requires_grad 为 T 的变量x b所计算得到的 新变量 t，也会被默认设置为 requires_grad=True 的
print(x.requires_grad, b.requires_grad, t.requires_grad)

'''
注意 torch 中的 autograd 机制存在一个 梯度累加的机制，因此在实际应用中 在每一轮的训练更新 计算梯度下降时，需要先将原来的梯度清零，见下节
'''