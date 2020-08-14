import torch.nn as nn
import os
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score,confusion_matrix
from scipy import stats
import torch.utils.data as Data
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# plt.rcdefaults()
from thop import profile
from module_uci_har import LegoConv2d
import sklearn.metrics as sm

from torch.autograd import Variable

from torchvision.models import resnet18





import pylab as pl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes






from sklearn.decomposition import PCA
from skimage import feature as ft #hog
GLOBAL_SEED = 1
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


torch.cuda.set_device(0)
n_gpu = torch.cuda.device_count()
# print(n_gpu)
path = os.path.dirname(os.path.abspath("__file__"))
# print(path)

pathlist = ['/home/tangyin/桌面/emnist/LegoNet-master/data_uci/uci_har/np_train_x.npy',
            '/home/tangyin/桌面/emnist/LegoNet-master/data_uci/uci_har/np_train_y.npy',
            '/home/tangyin/桌面/emnist/LegoNet-master/data_uci/uci_har/np_test_x.npy',
            '/home/tangyin/桌面/emnist/LegoNet-master/data_uci/uci_har/np_test_y.npy',
            ]



# # @torchsnooper.snoop()
def data_flat(data_y):
    data_y = np.argmax(data_y, axis=1)
    return data_y




def load_data(train_x_path, train_y_path, batchsize):
    train_x = np.load(train_x_path)
    train_x_shape = train_x.shape
    train_x = torch.from_numpy(
        np.reshape(train_x.astype(float), [train_x_shape[0], 1, train_x_shape[1], train_x_shape[2]])).cuda()

    train_y = data_flat(np.load(train_y_path))
    train_y = torch.from_numpy(train_y).cuda()
    print(train_y.shape)
    print(train_x.shape)

    torch_dataset = Data.TensorDataset(train_x, train_y)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=0,
    )
    total = len(loader)
    return loader




class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        print('this is loss function!')

    def forward(self, output, label):
        loss_func = F.cross_entropy(output, label)
        return loss_func


cfg = {'lego_vgg16': [64, 64, 64,  128, 128, 128, 256, 256, 256,  512, 512, 512,  512, 512, 512], }


class lego_vgg16(nn.Module):
    def __init__(self, vgg_name, n_split, n_lego, n_classes):
        super(lego_vgg16, self).__init__()
        self.n_split, self.n_lego, self.n_classes = n_split, n_lego, n_classes
        self.features = self._make_layers(cfg[vgg_name])  # 图像特征提取网络结构（只包含lego卷积和池化）
        self.classifier = nn.Linear(23040, n_classes,)

    def forward(self, x):
        # print(type(x),'type')
        # x = x.type(torch.cuda.DoubleTensor)
        x = self.features(x.float())
        # print(type(x), 'type')

        x = x.contiguous().view(x.size(0), -1)
        # x = self.linear(x)
        # x = self.out(x)
        # x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = x.double()
        return x

    def _make_layers(self, cfg):
        layers = []
        in_channels = 1
        list = [128, 256, 384]
        for i, x in enumerate(list):
            if i == 0:
                layers += [nn.Conv2d(in_channels, x, (6, 1), stride=(3, 1), padding=1),  # 第一层conv2d卷积操作
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True)]
                in_channels = x
                continue

            # if i == 0:
            #     layers += [LegoConv2d(in_channels, x, 3, self.n_split, self.n_lego),  # 第一层conv2d卷积操作
            #         nn.BatchNorm2d(x),
            #         nn.ReLU(inplace=True)]
            #     in_channels = x
            #     continue
            # if x == 'M':
            #     layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [LegoConv2d(in_channels, x, 6, self.n_split, self.n_lego),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
            # else:
            #     layers += [nn.Conv2d(in_channels, x, (6, 1), stride=(3, 1), padding=1),
            #                nn.BatchNorm2d(x),
            #                nn.ReLU(inplace=True)]
                in_channels = x
        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def copy_grad(self, balance_weight):
        for layer in self.features.children():
            if isinstance(layer, LegoConv2d):
                layer.copy_grad(balance_weight)


def to_one_hot(y, n_dims=None):
    ''' Take integer tensor y with n dims and convert it to 1-hot representation with n+1 dims. '''
    y_tensor = y.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return y_one_hot


def count_memory(model):
    total_params = 0
    for name, param in model.named_parameters():
        if 'aux' in name:
            continue
        total_params += np.prod(param.size())
    return total_params / 1e6


def adjust_learning_rate(optimizer, epoch):
  lr = 4e-4 * (0.1 ** (epoch // 100))
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr

def plot_confusion(comfusion,class_data):
    plt.figure(figsize=(9, 7))
    # plt.rcParams['figure.dpi'] = 1000
    plt.rcParams['font.family'] = ['Times New Roman']
    classes = class_data
    plt.imshow(comfusion, interpolation='nearest', cmap=plt.cm.Greens)  # 按照像素显示出矩阵
    plt.title('Confusion Matrix for UCI dataset', fontsize=6)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=315)
    plt.yticks(tick_marks, classes)
    plt.tick_params(labelsize=6)
    thresh = comfusion.max() / 2.
    # iters = [[i,j] for i in range(len(classes)) for j in range((classes))]
    # ij配对，遍历矩阵迭代器
    iters = np.reshape([[[i, j] for j in range(len(classes))] for i in range(len(classes))], (comfusion.size, 2))
    for i, j in iters:
        plt.text(j, i, format(comfusion[i, j]))  # 显示对应的数字

    plt.ylabel('Real Label', fontsize=6)
    plt.xlabel('Prediction Label', fontsize=6)

    plt.tight_layout()
    plt.savefig('/home/tangyin/桌面/emnist/LegoNet-master/data_uci/wisdm/wisdm_ccx.png', dpi=300)
    # plt.show()


def train(train_loader, test_x_path, test_y_path, train_error, test_error, optim,sch):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    loss_total_global = 0
    # print('Total_Number of params: {} |Trainable_num of params: {}'.format(total_num, trainable_num))
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
        # print(batch_x.shape, batch_y.shape)

        # statistical_x = get_statistical_feature(np.reshape(batch_x.cpu().numpy(), newshape=[-1, 128, 9]))
        # print(statistical_x==np.load(pathlist[4])[:200])
        target_onehot = to_one_hot(batch_y)
        target_onehot = target_onehot.cuda()
        # print(batch_x.shape,target_onehot.shape,batch_y.shape,'batch_x,target_onehot.shape,batch_y.shape')
        optimizer.zero_grad()
        output = model(batch_x)
        # print(output.shape,batch_y.shape,target_onehot.shape,'output.shape')

        loss = loss_func(output, batch_y.long())
        loss_total_global += loss.item() * batch_x.size(0)
        loss.backward()
        optimizer.step()
    train_output = torch.max(output, 1)[1].cuda()
    taccuracy = (torch.sum(train_output == batch_y.long()).type(torch.FloatTensor) / batch_y.size(0)).cuda()
    error = 1 - taccuracy.item()
    train_error.append(error)
        # print(train_error)
    if epoch % 1 == 0:
        model.eval()

        test_x = np.load(test_x_path)
        # statistical_xx=np.load(pathlist[5])
        test_x_shape = test_x.shape
        test_x = torch.from_numpy(np.reshape(test_x, [test_x_shape[0], 1, test_x_shape[1], test_x_shape[2]])).cuda()
        # statistical_xx = get_statistical_feature(np.reshape(test_x.cpu().numpy(), newshape=[-1, 128, 9]))
        # print(test_x.shape,statistical_x.shape)
        test_y = data_flat(np.load(pathlist[3]))
        test_y = torch.from_numpy(test_y).cuda()

        test_y_onehot = to_one_hot(test_y)
        test_y_onehot = test_y_onehot.cuda()
        print(test_y_onehot.shape)

        # print(test_x.shape, test_y.shape, test_y_onehot.shape, 'test_x.shape,test_y.shape,target_y.shape')
        try:
            test_output = model(test_x)
            test_output_copy = test_output
            # print(test_output.shape)
            test_output = data_flat(test_output.cpu().detach().numpy())
            # print(test_output.shape, 'test_output')

            test_output_f1 =np.asarray(pd.get_dummies(test_output))

            # print(test_y_onehot.shape, test_output_f1.shape)
            acc = accuracy_score(test_y_onehot.cpu().numpy(), test_output_f1)
            f1 = f1_score(test_y_onehot.cpu().numpy(), test_output_f1, average='weighted')
            f2 = f1_score(test_y_onehot.cpu().numpy(), test_output_f1, average='micro')
            f3 = f1_score(test_y_onehot.cpu().numpy(), test_output_f1, average='macro')
            reca = recall_score(test_y_onehot.cpu().numpy(), test_output_f1, average='weighted')
            print('Epoch: ', epoch, 'lr:', optimizer.param_groups[0]['lr'], '\n', '| test accuracy: %.8f' % acc, '| test F1: %.8f' % f1, '| test recall: %.8f' % reca, '| test micro: %.8f' % f2, '| test micro: %.8f' % f3)
            model.train()
        except ValueError:
            test_output = torch.max(test_output_copy, 1)[1].cuda()
            # print(test_output.shape,'test_output.shape')
            accuracy = (torch.sum(test_output == test_y.long()).type(torch.FloatTensor) / test_y.size(0)).cuda()
            # logging.info('\nEpoch: {}| ,test accuracy: {}'.format(epoch,accuracy))
            print('Epoch: ', epoch, '| test accuracy: %.8f' % accuracy)
            # test_error.append((1-accuracy.item()))
            print('error')
            model.train()
        else:
            pass
        test_output = torch.max(test_output_copy, 1)[1].cuda()
        accuracy = (torch.sum(test_output == test_y.long()).type(torch.FloatTensor) / test_y.size(0)).cuda()
        # print(type((1 - accuracy.item())))
        test_error.append((1 - accuracy.item()))
        # train_error.append((1 - accuracy.item()))
        # np.save("/home/tangyin/桌面/emnist/LegoNet-master/data_uci/uci_har/com8x_loss.npy", test_error)
        # np.save("/home/tangyin/桌面/emnist/LegoNet-master/data_uci/uci_har/tr_loss.npy", train_error)
        # print(test_loss.shape)
        if epoch % 100 == 0:
            confusion = sm.confusion_matrix(test_y.cpu().numpy(), test_output.cpu())
            print('混淆矩阵为：', confusion, sep='\n')
            plot_confusion(confusion,
                           ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"])



if __name__ == '__main__':

    model = lego_vgg16('lego_vgg16', 2, 0.5, 6)
    model.cuda()
    print(model)
    print('\n')
    print('Memory:', count_memory(model))
    # input = torch.randn(1, 1, 128, 9)
    # input = input.cuda()
    # flops, params = profile(model, inputs=(input,))
    # print('FLOPs:', flops)
    # print('\n')
    # model.eval()
    # input_tensor = torch.rand(1, 1, 128, 9)
    # script_model = torch.jit.trace(model, input_tensor.cuda())
    # script_model.save("/home/tangyin/桌面/emnist/LegoNet-master/data_uci/uci_har/uci_har.pt")
    # model.load_state_dict(torch.load('/home/tangyin/桌面/emnist/LegoNet-master/data_uci/uci_har/uci_har.pth'))

    # Export the trained model to ONNX
    # dummy_input = Variable(torch.randn(1, 1, 128, 9))  # one black and white 28 x 28 picture will be the input to the model
    # torch.onnx.export(model, dummy_input.cuda(), "/home/tangyin/桌面/emnist/LegoNet-master/data_uci/uci_har/uci_har.onnx")



    # params = list(model.parameters())
    # k = 0
    # for i in params:
    #     l = 1
    #     print("该层的结构：" + str(list(i.size())))
    #     for j in i.size():
    #         l *= j
    #     print("该层参数和：" + str(l))
    #     k = k + l
    # print("总参数数量和：" + str(k))
    # print('\n')


    # print('Params:', params)


    # optimizer = torch.optim.SGD(model.parameters(), lr=5e-4, momentum=0.9, weight_decay=0.0005)
    optimizer = torch.optim.Adam(model.parameters(), lr=4e-4)

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=4e-4, alpha=0.9)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10)
    scheduler = 1
    loss_func = nn.CrossEntropyLoss().cuda()
    train_loader = load_data(pathlist[0], pathlist[1], batchsize=200)
    train_error = []
    test_error = []
    max_accuracy = 0.0
    for epoch in range(500):
        train(train_loader, pathlist[2], pathlist[3], train_error, test_error, optimizer, scheduler)
        adjust_learning_rate(optimizer, epoch)

    # torch.save(model.state_dict(), '/home/tangyin/桌面/emnist/LegoNet-master/data_uci/uci_har/uci_har.pth')



    # test_error = np.load("/home/tangyin/桌面/emnist/LegoNet-master/data_uci/uci_har/com16x_loss.npy")
    # train_error = np.load("/home/tangyin/桌面/emnist/LegoNet-master/data_uci/uci_har/tr_loss.npy")
    # x = te_loss[:, 0]
    # y = te_loss[:, 1]
    # x1 = data2_loss[:, 0]
    # # y1 = data2_loss[:, 1]
    # fig = plt.figure(figsize=(7, 5))  # figsize是图片的大小`
    # ax1 = fig.add_subplot(1, 1, 1)  # ax1是子图的名字`
    #
    # pl.plot(x, y, 'g-', label=u'Dense_Unet(block layer=5)')
    # # ‘’g‘’代表“green”,表示画出的曲线是绿色，“-”代表画的曲线是实线，可自行选择，label代表的是图例的名称，一般要在名称前面加一个u，如果名称是中文，会显示不出来，目前还不知道怎么解决。
    # # p2 = pl.plot(x1, y1, 'r-', label=u'RCSCA_Net')
    # pl.legend()
    # # 显示图例
    # # p3 = pl.plot(x2, y2, 'b-', label=u'SCRCA_Net')
    # pl.legend()
    # pl.xlabel(u'iters')
    # pl.ylabel(u'loss')
    # plt.title('Compare loss for different models in training')

    # epochs = range(len(test_error))
    # epochs = np.arange(1)


    # plt.plot(epochs, acc, 'bo', label='Training acc')  # 'bo'为画蓝色圆点，不连线
    # plt.plot(epochs, val_acc, 'b', label='Validation acc')
    # plt.title('Training and validation accuracy')
    # plt.legend()  # 绘制图例，默认在右上角

    # plt.figure()
    #
    # plt.plot(epochs, test_error, 'r-', label='Test Loss')
    # plt.plot(epochs, train_error, 'b-', label='Train Loss')
    # plt.plot(epochs, val_loss, 'b', label='Validation loss')
    # plt.title('Training and validation loss')
    # pl.xlabel(u'Epoch')
    # pl.ylabel(u'Test Loss')
    # plt.legend()
    # plt.show()
    # fig_dir = '/home/tangyin/桌面/emnist/LegoNet-master/data_uci/uci_har'
    # fig_ext = '.png'
    # plt.savefig(os.path.join(fig_dir, 'LegoConv2d' + fig_ext),
    #             bbox_inches='tight', pad_inches=0)
    # # plt.savefig('/home/tangyin/桌面/emnist/LegoNet-master/data_uci/uci_har/uci_har.jpg')
    #



