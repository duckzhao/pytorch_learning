import torch
from torchvision import transforms, datasets, models
import os
import json
import numpy as np
from matplotlib import pyplot as plt
from torch import nn
import time
import copy

# 1.数据读取以及预处理
data_dir = './flower_data/'
train_data_dir = data_dir + '/train'
valid_data_dir = data_dir + '/valid'

# 使用 torchvision 定义 在训练过程中需要对 图片数据的预处理（数据增强部分）
# data_transforms中指定了所有图像预处理操作，似乎transforms 采用的是 PIL 读取的数据，并传输的数据流
data_transforms = {
    'train': transforms.Compose([
        # 然后按顺序在list中一行一行的写预处理/数据增强的操作，类似于tf2中的sequential结构的定义
        transforms.Resize((256, 256)),    # 对图片进行resize操作
        transforms.RandomRotation(45),  # 指定对图片进行 正负45°的随机旋转
        transforms.CenterCrop(224),    # 中心剪裁，将图像从中心为原点剪裁为 224，224的大小
        transforms.RandomHorizontalFlip(p=0.5),     # 图像随机反转
        transforms.RandomVerticalFlip(p=0.5),   # 图像垂直反转
        transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),  # 随机的对图片的亮度、对比度等进行调整
        transforms.RandomGrayscale(p=0.025),    # 随机的将图片转为灰度-但依旧保持三通道，只是三个通道的像素数值都相等
        transforms.ToTensor(),  # 将前面处理的结果转为tensor格式，推断之前的操作可能底层基于PIL ndr等其他库完成，返回值非tensor格式
        # 对处理后的图像完成 标准化变换 的操作，这个值，主要是取决于我们迁移学习时/hub 加载的model规定的值，因为他们的model在训练的时候对
        # 原始图像进行了这个尺度的处理，所以我们也要进行这个尺度的处理才能更好的拟合迁移学习的model，尤其是要pretrain时！
        # 这个Normalize对应在np等库的操作实际上就是 （x-[0.485, 0.456, 0.406]）/[0.229, 0.224, 0.225],具体的权重参数可以在该网站寻得：
        # https://pytorch.org/vision/stable/models.html 似乎都是同一个值
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 均值，标准差
    ]),
    # 验证时进行的预处理操作无需和训练时保持一致，无需进行数据增强部分，仅保证送入的大小，归一化等 操作一致即可
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

# 利用 torchvison 提供的视觉处理库 中的 datasets 和 torch.utils.data.DataLoader 完成对图片批训练 数据加载器 dataloaders 的生成
batch_size = 32
# datasets.ImageFolder() 需要保证当前 数据集 是以 按训练集、测试集 以及下一级的类别都分类好的 实际图片 保存形式，标签名是文件夹名称这样子
# 执行完后 返回的 dataset 拥有 classes ->用一个 list 保存类别名称，class_to_idx imgs三个属性，
# 它的结构就是[(img_data,class_id),(img_data,class_id),…]，以及可以使用dataset[0]查看读入处理后的图片结果
image_datasets = {x: datasets.ImageFolder(root=os.path.join(data_dir, x), transform=data_transforms[x]) for x in ['train', 'valid']}
class_names = image_datasets['train'].classes
# 注意 dataset 给图片类别进行的 数字编码 是随机数字编码，和类别文件夹的 名称 无直接关联，需要通过 class_to_idx 把 类别和随机索引对照起来
# print(image_datasets['train'].class_to_idx)

# print(image_datasets['train'][0])
# print({x: len(image_datasets[x]) for x in ['train', 'valid']})
# 接下来利用 torch.utils.data.DataLoader 按照batch_size生成 数据集加载器,可能出现batch_size小于预期的情况，请指定drop_last = True解决
dataloaders = {x: torch.utils.data.DataLoader(dataset=image_datasets[x], batch_size=batch_size, shuffle=True,
                                              drop_last=True) for x in ['train', 'valid']}

# 根据json文件 读取文件夹名称对应的 实际标签
with open('./cat_to_name.json', 'r', encoding='utf-8') as f:
    cat_2_name = json.loads(f.read())
# print(cat_2_name)

# 利用 dataloaders 加载一次数据集，并绘制展示
def im_convert(tensor):
    '''
    展示使用dataloader夹杂的 tensor 图片数据
    :param tensor: 传入一个tensor格式的图片数据
    :return:
    '''
    # 训练时数据可能都在gpu里面，所以我们手动 拷贝一份 到cpu
    image = tensor.to('cpu').clone().detach()
    # 将数据转为np格式，并squeeze删除 冗余的维度 eg [[1 2 3]]->[1 2 3]
    image = image.data.numpy().squeeze()
    # 实际上是做了一个矩阵的转置，为了满足plt绘图时 hwc 的像素表示方法
    image = image.transpose(1, 2, 0)
    # 将标准化后的 tensor 转为标准化之前的数据，逆标准化 Normalize = (img-mean)/std -> tensor*std+mean
    image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    # 将 image 取值截断在0-1之间
    image = image.clip(0, 1)
    return image

# 使用 iter 获得一个 加载器对象的迭代器
dataiter = iter(dataloaders['valid'])
# 获取 batch size 张图
images, classes = dataiter.next()
# print(classes)

# 一共画8张图
for index in range(8):
    plt.subplot(2, 4, index+1)
    plt.imshow(im_convert(images[index]))
    plt.title(cat_2_name[str(int(class_names[classes[index]]))])
plt.show()

# 开始配置迁移学习的 加载设置，包括是否pretrain，是否训练全部层参数，是否使用gpu训练
model_name = 'resnet'   #可选的比较多 ['resnet', 'alexnet', 'vgg', 'squeezenet', 'densenet', 'inception']
# 是否使用预训练参数--->一般默认使用
use_pretrained = True
# 是否训练 pretrained 参数，True 表示 不对pretrained的参数进行训练，False表示对pretrained的参数进行二次训练，即记录求导梯度
feature_extract = True
# 测试能否使用gpu训练
train_on_gpu = torch.cuda.is_available()
if train_on_gpu:
    print('gpu is available, training on gpu')
else:
    print('gpu is not available, training on cpu')
# 指定device为gpu或者cpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 指定新模型的分类数---根据当前任务决定
num_classes = 102

def set_model_parameter_requires_grad(model, feature_extract):
    '''
    如果 指定使用预训练参数，则预训练的参数部分，即卷积层参数，无需反向传播优化，设置为不记录梯度,当然也不绝对，如果把新接的全连接层训练的差不多了，
    可以设置这些预训练层为记录梯度模式，整体再进行二次训练，取得最佳优化效果。---主要取决于样本数量
    :param model:
    :param feature_extract:ture表示使用预训练参数
    :return:指针变量，无需返回
    '''
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False

# 使用如下两句话可以打印 加载的网络结构
# model = models.resnet152(pretrained=True)
# print(model)

def init_model(model_name, num_classes, feature_extract, use_pretrained=True):
    '''
    根据输入返回配置好的 model 和 模型输入 图片大小size
    :param model_name:可选的比较多 ['resnet', 'alexnet', 'vgg', 'squeezenet', 'densenet', 'inception']
    :param num_classes:当前分类任务的类别数
    :param feature_extract:是否训练 pretrained 的参数
    :param use_pretrainrd:是否使用 pretrained 模型参数
    :return:
    '''
    # 防止程序报黄--if else流程，预定义返回值，实际没用
    model_ft = None
    input_size = 0
    if model_name == 'resnet':
        model_ft = models.resnet152(pretrained=use_pretrained)
        set_model_parameter_requires_grad(model_ft, feature_extract)
        # model_ft.fc 指的是 当前 model 中的 fc 层，即最后一层全连接层，fc 是该层的名称，可以在print(model_ft)中看到这个名称，包括其他
        # 层的名称，可以使用model.name去调用这些层，model_ft.fc.in_features表示 该全连接层的 输入特征，即Linear的in_features，因为
        # torch的全连接层Linear是需要指定输入维度的，和输出维度（全连接层神经元的个数），所以我们需要通过 model_ft.fc.in_features 手动
        # 获得当前模型针对 指定大小的输入图形 卷积处理后得到的 特征长度（in_features），以供我们指定适合当前分类的 Linear 层
        num_ftrs = model_ft.fc.in_features
        # 根据自己的任务需求，修改当前model的全连接层，新增的层，默认requires_grad=True
        model_ft.fc = torch.nn.Sequential(torch.nn.Linear(in_features=num_ftrs, out_features=num_classes),
                                          torch.nn.LogSoftmax(dim=1))
        # 模型预训练时规定的图像输入大小
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_model_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg16(pretrained=use_pretrained)
        set_model_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_model_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_model_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_model_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

model_ft, input_size = init_model(model_name, num_classes, feature_extract, use_pretrained)

# 给当前model配置gpu/cpu
model_ft = model_ft.to(device)

# 模型保存
checkpoint_filename = 'checkpoint.pth'

# 训练总轮数
epochs = 20

# 查看当前model的所有参数，但是这些参数在fine tuning的时候不一定全部都要传进去 优化器进行训练，如果 features_extract=T,则表示仅训练最后自己
# 设置的全连接层，如果 features_extract=F 则表示全部都要训练，所以下面对 送进 优化器的 params_to_update 做了一个判断赋值
# 查看当前配置下，模型需要学习的参数有哪些
print('parameters need to learn:')
if feature_extract:
    # 在 冻结预训练参数 情况下，通过model每层参数的 requires_grad 判断是否需要更新（是否需要传入优化器迭代），即仅训练新增的层
    params_to_update = []
    for name, param in model_ft.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
            print('\t', name)
else:
    # 否则，则训练全部参数
    params_to_update = model_ft.parameters()
    for name, param in model_ft.named_parameters():
        if param.requires_grad:
            print('\t', param)
# print(model_ft)

# 配置优化器，将需要训练的参数传入优化器，并指定学习率
optimizer_ft = torch.optim.Adam(params_to_update, lr=1e-2)
# 配置学习率下降方式，一开始学习率较大，随着epoch进行，学习率逐渐减小，每7个epoch衰减成原来的1/10---加快优化速度和效果
scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# 指定损失函数，仅用 NLLLoss 即可
# 最后一层已经LogSoftmax()了，所以不能nn.CrossEntropyLoss()来计算了，nn.CrossEntropyLoss()相当于logSoftmax()和nn.NLLLoss()整合
criterion = nn.NLLLoss()

def train_model(model, dataloaders, criterion, optimizer, scheduler, epochs, checkpoint_filename, is_incepption=False):
    since = time.time()
    # 记录当前model在验证集上最佳的准确率，用于保存最佳model到本地
    best_acc = 0
    # 如果之前训练过，可以通过判断 checkpoint文件在不在决定是否 续着训练
    if os.path.exists(checkpoint_filename):
        print('load the checkpoint!')
        checkpoint = torch.load(checkpoint_filename)
        model_ft.load_state_dict(checkpoint['state_dict'])

        # 因为此时我们训练了所有层，所以当前 优化器中待训练的参数 和 checkpoint中的 数量是不一致的，此时无法加载优化器，只能重新设置一个优化器
        # optimizer.load_state_dict(checkpoint['optimizer'])

        best_acc = checkpoint['best_acc']
        # model.class_to_idx = checkpoint['mapping']

        # 如果不是加载历史的 model的优化器 ，optimizer.param_group[0]['lr']为空会报错
        LRs = [optimizer.param_group[0]['lr']]

        print(best_acc, LRs)

    model.to(device)
    # 记录训练中每个epochs的 acc 和 loss 以便观察
    train_acc_history = []
    val_acc_history = []
    train_losses = []
    val_losses = []
    # 可以看为用于记录优化器学习率变化的列表，如果不是加载历史的model ，optimizer.param_group[0]['lr']为空会报错
    # LRs = [optimizer.param_group[0]['lr']]
    LRs = []

    # 预定义一个当前最佳model的内存对象---在训练循环中更替
    best_model_wts = copy.deepcopy(model.state_dict())

    # 开始进入正式训练流程
    for epoch in range(epochs):
        print(f'epoch {epoch+1}/{epochs}')
        print('-'*10)

        # 每一个epoch 都包括 train 和 valid 两个过程，指定train或者eval 主要是为了BN和Dropout层 在训练和测试 时候有所不同，需要说明
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()   # 训练
            else:
                model.eval()    # 评估

            # 用于统计每个 训练 epoch 中的 loss 和 corrects
            running_loss = 0.0
            runing_corrects = 0.0

            # 开始进入真实的取数据-训练 循环
            # 使用for循环 从 dataloader 中取数据，每次取出指定的一个 batch size
            for inputs, labels in dataloaders[phase]:
                # tensor([76, 40, 77, 36, 85, 54, 11, 50, 88, 47, 40, 69, 26, 89, 53, 59, 74, 77,
                #         94, 68, 38, 74, 77, 29, 18,  7, 33, 15, 25, 42, 32, 57])
                # print(labels)

                # 将输入转到gpu中
                inputs = inputs.to(device)
                labels = labels.to(device)
                # 梯度清零
                optimizer.zero_grad()

                # 计算梯度并反向传播---->尽在训练时进行
                # torch.set_grad_enabled(mode) 当 mode=True 时对with下的操作记录梯度，否则不记录梯度，训练时开始记录梯度，验证时=False
                with torch.set_grad_enabled(phase == 'train'):
                    # incepption net 需要走这个流程，它比较特殊
                    if is_incepption and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    # 其余经典网络都走这个流程
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    # 对 outputs进行如下转换得到preds，才能和 labels的形式对应起来
                    _, preds = torch.max(outputs, 1)
                    # tensor([17, 69, 97, 37,  6, 96, 64, 54, 41, 54, 90, 37,  7, 86, 12, 44, 76, 65,
                    #         96, 30, 15, 89, 72, 40, 12,  3, 99, 78, 96,  7, 96, 47], device='cuda:0'))
                    # print(preds)

                    # 仅在训练阶段更新权重
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 累加每个 batch 的 损失
                running_loss += loss.item() * inputs.size(0)
                runing_corrects += torch.sum(preds == labels.data)

            # 迭代数据的for循环结束，标志着一个epoch训练结束，统计该epoch的平均 loss和acc
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = runing_corrects / len(dataloaders[phase].dataset)

            # 打印当前epoch的训练时间和准确率，损失的信息
            time_elapsed = time.time() - since
            print('Time elapsed {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # 每个epoch中 都需要在 valid 验证结束后，根据 valid 的 loss 和 acc 判断当前model是否当前最佳，并保存当前最佳的model
            if phase == 'valid' and epoch_acc > best_acc:
                # 更新最佳的 model 准确率
                best_acc = epoch_acc
                # 更新最佳的 model 训练参数 到 内存中，以便在训练结束后，直接从内存中加载最佳的model，而不用再从 checkpoint文件中去读取
                best_model_wts = copy.deepcopy(model.state_dict())
                # state中保存了完整的当前 model 的 checkpoint，主要用于以后恢复当前训练点，进行继续训练。
                # 如果只想保存当前model，以后用于预测任务，则仅需保存 model.state_dict()/best_model_wts 即可
                state = {
                    'state_dict': best_model_wts,   # model 每层权重参数
                    'best_acc': best_acc,   # 当前验证最佳准确率
                    'optimizer': optimizer.state_dict()     # 当前训练过程中 优化器的参数
                }
                torch.save(state, checkpoint_filename)

            # 记录每个 epoch 中的 train 的 acc 和 loss 变化数值，用于可视化训练信息
            if phase == 'train':
                # 更新 优化器的 学习率，仅在训练的epoch更新
                scheduler.step()
                train_acc_history.append(epoch_acc)
                train_losses.append(epoch_loss)
            if phase == 'valid':
                val_acc_history.append(epoch_acc)
                val_losses.append(epoch_loss)

        # 当每个 epoch 训练完成后，记录当前epoch 优化器的学习率
        print('Optimizer learning rate : {:.7f}'.format(optimizer.param_groups[0]['lr']))
        LRs.append(optimizer.param_groups[0]['lr'])
        print()

    # 当所有 epochs 训练完成后，打印训练花费的整体时间和 epoch最佳准确率
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 模型训练最后一轮的权重不一定是最佳权重，因此需要手动设置 训练过程中最佳的 权重 到model中
    model.load_state_dict(best_model_wts)

    return model, val_acc_history, val_losses, train_acc_history, train_losses, LRs

# 开始训练 --- 仅训练最后新增的全连接层
model_ft, val_acc_history, val_losses, train_acc_history, train_losses, LRs = train_model(model_ft, dataloaders,
                                                        criterion, optimizer_ft, scheduler, epochs, checkpoint_filename)
