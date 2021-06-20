import torch
from torchvision import models, transforms, datasets
import time
import copy
import os

# 复现网络结构，因为 模型上次训练的时候 用的是 pretrained 的model，所以这里加载模型的时候 是否应该 加上pretrained=True 的参数呢？
# 也就是说，上一轮我在 save model 的时候，因为预训练网络都被冻住了，没有参与训练，可能我保存的权重里，仅有最后一层权重，如果不 pretrained=True
# 的话，可能恢复出来的model前面的卷积层是没有参数的，这导致 当前model的 acc 小于 保存时的 acc。且这次训练后保存的权重文件大于第一次训练的文件大小
model_ft = models.resnet152()
num_ftrs = model_ft.fc.in_features
classes_num = 102
model_ft.fc = torch.nn.Sequential(torch.nn.Linear(num_ftrs, classes_num),
                                  torch.nn.Softmax(dim=1))
# 重新加载的model，所有层都是可训练的---加载权重不会改变 requires_grad
# for param in model_ft.parameters():
#     print(param.requires_grad)

checkpoint_filename = 'checkpoint.pth'

# 接着上次训练的model 保存点，继续训练 model 和 optimizer
# --->如果当前model中的待训练层有所冻结或者释放，则优化器无法恢复，只能从0开始重新训练，model weights可以恢复
optimizer = torch.optim.Adam(model_ft.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# 损失函数
criterion = torch.nn.NLLLoss()

# 查看gpu是否可用
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train_model(model, dataloaders, criterion, optimizer, epochs, checkpoint_filename, is_incepption=False):
    since = time.time()
    # 记录当前model在验证集上最佳的准确率，用于保存最佳model到本地
    best_acc = 0

    # 记录训练中每个epochs的 acc 和 loss 以便观察
    train_acc_history = []
    val_acc_history = []
    train_losses = []
    val_losses = []
    # 可以看为用于记录优化器学习率变化的列表
    LRs = []

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
        # LRs = [optimizer.param_group[0]['lr']]

        print(best_acc, LRs)

    model.to(device)

    # 预定义一个当前最佳model的内存对象---在训练循环中更替
    best_model_wts = copy.deepcopy(model.state_dict())

    # 开始进入正式训练流程
    for epoch in range(epochs):
        print(f'epoch {epoch + 1}/{epochs}')
        print('-' * 10)

        # 每一个epoch 都包括 train 和 valid 两个过程，指定train或者eval 主要是为了BN和Dropout层 在训练和测试 时候有所不同，需要说明
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # 训练
            else:
                model.eval()  # 评估

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
                    'state_dict': best_model_wts,  # model 每层权重参数
                    'best_acc': best_acc,  # 当前验证最佳准确率
                    'optimizer': optimizer.state_dict()  # 当前训练过程中 优化器的参数
                }
                torch.save(state, checkpoint_filename)

            # 记录每个 epoch 中的 train 的 acc 和 loss 变化数值，用于可视化训练信息
            if phase == 'train':
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


train_model(model_ft, dataloaders, criterion, optimizer, epochs=20, checkpoint_filename=checkpoint_filename)