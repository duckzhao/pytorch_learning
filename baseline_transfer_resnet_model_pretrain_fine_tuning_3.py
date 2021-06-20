import torch
from torchvision import models
import cv2
from matplotlib import pyplot as plt
import numpy as np
import json

# 恢复保存点
checkpoint_path = 'checkpoint.pth'
checkpoint = torch.load(checkpoint_path)

# 查看gpu状态
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = models.resnet152()
num_ftrs = model.fc.in_features
classes_num = 102
model.fc = torch.nn.Sequential(torch.nn.Linear(num_ftrs, classes_num),
                               torch.nn.Softmax(dim=1))

model = model.to(device)
model.load_state_dict(checkpoint['state_dict'])

best_acc = checkpoint['best_acc']
print(best_acc)

# 使用训练好的model预测数据时，需要对预测的数据进行 和 valid 一样的预处理，即valid的 transforms.Compose 操作
# 包括：输入的大小size是一致、归一化、标准化操作等等
# 传入一个图片地址，返回处理好的，能够直接送进model预测的 tensor
def process_image(image_path):
    img = cv2.imread(image_path)
    # 由于cv2 读取的图片 颜色通道顺序为 BGR，所以我们需要转换为RBG，以保持和 model 训练时 一致（torch以PIL读取图片，格式为RGB）
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # resize 操作
    img = cv2.resize(img, (256, 256))
    # crop 操作
    img = img[int((256-224)/2): int((256-224)/2)+224, int((256-224)/2): int((256-224)/2)+224]
    plt.imshow(img)
    plt.show()
    # （但是需要注意一点训练数据是在0-1上进行标准化，所以我们多了一步归一化操作）归一化、标准化操作
    img = img / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    # 转换 矩阵 像素排列顺序，以符合torch输入,   H x W x C ->（nSample）x C x H x W
    img = np.transpose(img, (2, 0, 1))
    # 新增一个batch 的维度,以维持 批训练 的一致性
    img = img[np.newaxis, :]
    img = torch.tensor(img)
    return img

img = process_image('./test.jpg')
# 指定当前为预测模式
model.eval()
# img.type(torch.FloatTensor)转换 tensor 类型从double为float，.cuda() 将预测tensor传到 cuda上（因为model在cuda上已经）,利用gpu资源
output = model(img.type(torch.FloatTensor).cuda())
# torch.max(output, dim=1)[0] 返回的是概率， [1]返回的是类别index，这个index顺序和 文件夹顺序类别 保持一致，可以一一对应回去
with open('./cat_to_name.json', 'r', encoding='utf-8')as f:
    cat_2_name = json.loads(f.read())

# 由于 torch.max(output, dim=1)[1].item() 拿到的标签是 datasets.ImageFolder 自己根据原始文件数据集名称索引的数字编码，所以需要将这个
# 数字预测结果返回值 和 原始的label对应起来，需要用到 datasets.classes_name属性，所以建议保存checkpoint时，连同 classes_name 一起保存
# 以便在使用模型进行预测时，恢复 索引和真实标签 对照关系
classes_name = ['1', '10', '100', '101', '102', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '4', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '5', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '6', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '7', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '8', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '9', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99']
print(classes_name[torch.max(output, dim=1)[1].item()], cat_2_name[classes_name[torch.max(output, dim=1)[1].item()]])