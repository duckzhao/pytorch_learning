'''
hub 实际上就是 tf 和 keras 中的 applications，将经典模型作为迁移学习和预训练模型进行导入,详细的模型可以在
https://pytorch.org/hub/ 中进行查看，包括模型的介绍，和使用方法说明都有详细的讲解
'''
import torch

# 使用过程中会先去torch的官网下载相应的model和权重文件
model = torch.hub.load('pytorch/vision:v1.8.1', 'alexnet', pretrained=True)
print(model.eval())

# 此时得到的model 就是一个 实例化好的，拥有权重的 model 了，可以按照规则给里面传入 数据进行预测，或者再进行微调的迁移学习了
# with torch.no_grad():
#     output = model(input_batch)