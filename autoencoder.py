import random

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as fun
import torch.optim as optim


import train_class

# 图片化方案
class AutoCoder2(nn.Module):
    def __init__(self, device, batch_size = 16) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.conv11 = nn.Conv2d(kernel_size=3, stride=1, in_channels=1, out_channels=16, padding=1).to(device)
        self.pool11 = nn.MaxPool2d(stride=4, kernel_size=4, return_indices=True).to(device)
        self.bn11 = nn.BatchNorm2d(16).to(device)
        self.conv12 = nn.Conv2d(kernel_size=3, stride=1, in_channels=16, out_channels=32, padding=1).to(device)
        self.pool12 = nn.MaxPool2d(stride=4, kernel_size=4, return_indices=True).to(device)
        self.bn12 = nn.BatchNorm2d(32).to(device)
        self.conv13 = nn.Conv2d(kernel_size=3, stride=1, in_channels=32, out_channels=64, padding=1).to(device)
        self.pool13 = nn.MaxPool2d(stride=4, kernel_size=4, return_indices=True).to(device)
        self.bn13 = nn.BatchNorm2d(64).to(device)
        self.conv14 = nn.Conv2d(kernel_size=3, stride=1, in_channels=64, out_channels=128, padding=1).to(device)
        self.pool14 = nn.MaxPool2d(stride=4, kernel_size=4, return_indices=True).to(device)
        self.bn14 = nn.BatchNorm2d(128).to(device)

        self.dnn11 = nn.Linear(2048, 128).to(device)
        self.dnn12 = nn.Linear(128, 16).to(device)

        self.dnn22 = nn.Linear(16, 128).to(device)
        self.dnn21 = nn.Linear(128, 2048).to(device)

        self.conv24 = nn.ConvTranspose2d(kernel_size=3, stride=1, in_channels=128, out_channels=64, padding=1).to(device)
        self.pool24 = nn.MaxUnpool2d(stride=4, kernel_size=4).to(device)
        self.bn24 = nn.BatchNorm2d(64).to(device)
        self.conv23 = nn.ConvTranspose2d(kernel_size=3, stride=1, in_channels=64, out_channels=32, padding=1).to(device)
        self.pool23 = nn.MaxUnpool2d(stride=4, kernel_size=4).to(device)
        self.bn23 = nn.BatchNorm2d(32).to(device)
        self.conv22 = nn.ConvTranspose2d(kernel_size=3, stride=1, in_channels=32, out_channels=16, padding=1).to(device)
        self.pool22 = nn.MaxUnpool2d(stride=4, kernel_size=4).to(device)
        self.bn22 = nn.BatchNorm2d(16).to(device)
        self.conv21 = nn.ConvTranspose2d(kernel_size=3, stride=1, in_channels=16, out_channels=1, padding=1).to(device)
        self.pool21 = nn.MaxUnpool2d(stride=4, kernel_size=4).to(device)
        self.bn21 = nn.BatchNorm2d(1).to(device)


    def forward(self, x):
        x, indices1 = self.pool11(fun.relu(self.bn11(self.conv11(x))))
        x, indices2 = self.pool12(fun.relu(self.bn12(self.conv12(x))))
        x, indices3 = self.pool13(fun.relu(self.bn13(self.conv13(x))))
        x, indices4 = self.pool14(fun.relu(self.bn14(self.conv14(x))))

        x = x.reshape(self.batch_size, 2048)

        x = fun.relu(self.dnn11(x))
        x = fun.relu(self.dnn12(x))

        # ========================

        x = fun.relu(self.dnn22(x))
        x = fun.relu(self.dnn21(x))

        x = x.reshape(self.batch_size, 128, 4, 4)

        x = fun.relu(self.bn24(self.conv24(self.pool24(x, indices4))))
        x = fun.relu(self.bn23(self.conv23(self.pool23(x, indices3))))
        x = fun.relu(self.bn22(self.conv22(self.pool22(x, indices2))))
        x = fun.relu(self.bn21(self.conv21(self.pool21(x, indices1))))

        return x

class AutoCoder1(nn.Module):
    def __init__(self, device) -> None:
        super().__init__()
        self.conv11 = nn.Conv1d(kernel_size=25, stride=1, in_channels=1, out_channels=16, padding=12).to(device)
        self.pool11 = nn.MaxPool1d(stride=25, kernel_size=25, return_indices=True).to(device)
        self.bn11 = nn.BatchNorm1d(16).to(device)
        self.conv12 = nn.Conv1d(kernel_size=25, stride=1, in_channels=16, out_channels=32, padding=12).to(device)
        self.pool12 = nn.MaxPool1d(stride=25, kernel_size=25, return_indices=True).to(device)
        self.bn12 = nn.BatchNorm1d(32).to(device)
        self.conv13 = nn.Conv1d(kernel_size=25, stride=1, in_channels=32, out_channels=64, padding=12).to(device)
        self.pool13 = nn.MaxPool1d(stride=25, kernel_size=25, return_indices=True).to(device)
        self.bn13 = nn.BatchNorm1d(64).to(device)
        self.conv14 = nn.Conv1d(kernel_size=25, stride=1, in_channels=64, out_channels=128, padding=12).to(device)
        self.pool14 = nn.MaxPool1d(stride=25, kernel_size=25, return_indices=True).to(device)
        self.bn14 = nn.BatchNorm1d(128).to(device)

        self.dnn11 = nn.Linear(2304, 256).to(device)
        self.dnn12 = nn.Linear(256, 16).to(device)

        self.dnn21 = nn.Linear(16, 256).to(device)
        self.dnn22 = nn.Linear(256, 2304).to(device)

        self.conv21 = nn.ConvTranspose1d(kernel_size=25, stride=1, in_channels=128, out_channels=64, padding=12).to(device)
        self.pool21 = nn.MaxUnpool1d(stride=25, kernel_size=25).to(device)
        self.bn21 = nn.BatchNorm1d(64).to(device)
        self.conv22 = nn.ConvTranspose1d(kernel_size=25, stride=1, in_channels=64, out_channels=32, padding=12).to(device)
        self.pool22 = nn.MaxUnpool1d(stride=25, kernel_size=25).to(device)
        self.bn22 = nn.BatchNorm1d(32).to(device)
        self.conv23 = nn.ConvTranspose1d(kernel_size=25, stride=1, in_channels=32, out_channels=16, padding=12).to(device)
        self.pool23 = nn.MaxUnpool1d(stride=25, kernel_size=25).to(device)
        self.bn23 = nn.BatchNorm1d(16).to(device)
        self.conv24 = nn.ConvTranspose1d(kernel_size=25, stride=1, in_channels=16, out_channels=1, padding=12).to(device)
        self.pool24 = nn.MaxUnpool1d(stride=25, kernel_size=25).to(device)
        self.bn24 = nn.BatchNorm1d(1).to(device)

    def forward(self, x):

        x, indices1 = self.pool11(fun.relu(self.bn11(self.conv11(x))))
        x, indices2 = self.pool12(fun.relu(self.bn12(self.conv12(x))))
        x, indices3 = self.pool13(fun.relu(self.bn13(self.conv13(x))))
        x, indices4 = self.pool14(fun.relu(self.bn14(self.conv14(x))))
        x = x.view(-1, 1, 128 * 18)
        x = fun.relu(self.dnn11(x))
        x = fun.relu(self.dnn12(x))

        x = fun.relu(self.dnn21(x))
        x = fun.relu(self.dnn22(x))

        x = x.view(-1, 128, 18)

        x = fun.relu(self.bn21(self.conv21(self.pool21(x, indices4))))
        x = fun.relu(self.bn22(self.conv22(self.pool22(x, indices3))))
        x = fun.relu(self.bn23(self.conv23(self.pool23(x, indices2))))
        x = fun.relu(self.bn24(self.conv24(self.pool24(x, indices1))))

        # x = fun.relu(self.dnn13(x))
        # x, indices1 = self.pool11(fun.relu(self.bn11(self.conv11(x))))
        # x, indices2 = self.pool12(fun.relu(self.bn12(self.conv12(x))))
        # x, indices3 = self.pool13(fun.relu(self.bn13(self.conv13(x))))
        #
        # x = x.reshape(118, -1)
        #
        # x = fun.relu(self.dnn11(x))
        # x = fun.relu(self.dnn12(x))
        # x = fun.relu(self.dnn13(x))
        #
        # # ========================
        #
        # x = fun.relu(self.dnn21(x))
        # x = fun.relu(self.dnn22(x))
        # x = fun.relu(self.dnn23(x))
        #
        # x = x.reshape(118, 128, 8, 8)
        #
        # x = fun.relu(self.bn21(self.conv21(self.pool21(x, indices3))))
        # x = fun.relu(self.bn22(self.conv22(self.pool22(x, indices2))))
        # x = fun.relu(self.bn23(self.conv23(self.pool23(x, indices1))))

        return x

# 参数太多，该方法不可行
class AutoCoderDnn(nn.Module):
    def __init__(self, device, drop=None) -> None:
        super().__init__()
        if drop is None:
            drop = [0, 0, 0, 0, 0, 0, 0, 0]

        self.device = device
        self.dnn11 = nn.Linear(7062985, 8192).to(device)
        self.dnn12 = nn.Linear(8192, 1024).to(device)
        self.dnn13 = nn.Linear(1024, 256).to(device)
        self.dnn14 = nn.Linear(256, 16).to(device)
        self.dnn24 = nn.Linear(16, 256).to(device)
        self.dnn23 = nn.Linear(256, 1024).to(device)
        self.dnn22 = nn.Linear(1024, 8192).to(device)
        self.dnn21 = nn.Linear(8192, 7062985).to(device)

        self.drop11 = nn.Dropout(drop[0]).to(device)
        self.drop12 = nn.Dropout(drop[1]).to(device)
        self.drop13 = nn.Dropout(drop[2]).to(device)
        self.drop14 = nn.Dropout(drop[3]).to(device)
        self.drop24 = nn.Dropout(drop[4]).to(device)
        self.drop23 = nn.Dropout(drop[5]).to(device)
        self.drop22 = nn.Dropout(drop[6]).to(device)
        self.drop21 = nn.Dropout(drop[7]).to(device)

    # 训练
    def forward(self, x):
        x = fun.relu(self.dnn11(x))
        x = self.drop11(x)
        x = fun.relu(self.dnn12(x))
        x = self.drop12(x)
        x = fun.relu(self.dnn13(x))
        x = self.drop13(x)
        x = fun.relu(self.dnn14(x))
        x = self.drop14(x)
        #
        x = fun.relu(self.dnn24(x))
        x = self.drop24(x)
        x = fun.relu(self.dnn23(x))
        x = self.drop23(x)
        x = fun.relu(self.dnn22(x))
        x = self.drop22(x)
        x = fun.relu(self.dnn21(x))
        x = self.drop21(x)
        #
        return x

    # 执行
    def forward2(self, x):

        x = fun.relu(self.dnn11(x))
        x = fun.relu(self.dnn12(x))
        x = fun.relu(self.dnn13(x))
        x = fun.relu(self.dnn14(x))

        return x


class Encoder(nn.Module):
    def __init__(self, device) -> None:
        super().__init__()
        # self.conv1 = nn.Conv2d(kernel_size=3, stride=1, in_channels=3, out_channels=32, padding=1).to(device)
        # self.pool1 = nn.MaxPool2d(stride=2, kernel_size=2, return_indices=True).to(device)
        # self.bn1 = nn.BatchNorm2d(32).to(device)
        # self.drop1 = nn.Dropout(0.7).to(device)
        # self.conv2 = nn.Conv2d(kernel_size=3, stride=1, in_channels=32, out_channels=64, padding=1).to(device)
        # self.pool2 = nn.MaxPool2d(stride=2, kernel_size=2, return_indices=True).to(device)
        # self.bn2 = nn.BatchNorm2d(64).to(device)
        # self.drop2 = nn.Dropout(0.7).to(device)
        # self.conv3 = nn.Conv2d(kernel_size=3, stride=1, in_channels=64, out_channels=128, padding=1).to(device)
        # self.pool3 = nn.MaxPool2d(stride=2, kernel_size=2, return_indices=True).to(device)
        # self.bn3 = nn.BatchNorm2d(128).to(device)
        # self.drop3 = nn.Dropout(0.7).to(device)
        # self.conv4 = nn.Conv2d(kernel_size=3, stride=2, in_channels=128, out_channels=256, padding=1).to(device)
        # self.pool4 = nn.MaxPool2d(stride=1, kernel_size=2, return_indices=True).to(device)
        # self.bn4 = nn.BatchNorm2d(256).to(device)
        # self.drop4 = nn.Dropout(0.7).to(device)

        self.dnn1 = nn.Linear(4096, 1024).to(device)
        # self.bn1 = nn.BatchNorm1d(12800).to(device)
        self.drop1 = nn.Dropout(0.7).to(device)
        self.dnn2 = nn.Linear(1024, 512).to(device)
        self.bn2 = nn.BatchNorm1d(8192).to(device)
        self.drop2 = nn.Dropout(0.7).to(device)
        self.dnn3 = nn.Linear(512, 256).to(device)
        self.bn3 = nn.BatchNorm1d(4096).to(device)
        self.drop3 = nn.Dropout(0.7).to(device)

    def forward(self, x):
        # x = self.conv1(x)
        # x = fun.relu(self.bn1(x))
        # x, indices1 = self.pool1(x)
        # x = self.drop1(x)
        #
        # x = self.conv2(x)
        # x = fun.relu(self.bn2(x))
        # x, indices2 = self.pool2(x)
        # x = self.drop2(x)
        #
        # x = self.conv3(x)
        # x = fun.relu(self.bn3(x))
        # x, indices3 = self.pool3(x)
        # x = self.drop3(x)

        # x = self.conv4(x)
        # x = fun.relu(self.bn4(x))
        # x, indices4 = self.pool4(x)
        # x = self.drop4(x)

        x = x.reshape(1, -1)
        x = fun.relu(self.dnn1(x))
        # x = self.bn1(x)
        # x = self.drop1(x)
        # x = fun.relu(self.dnn2(x))
        # x = self.bn2(x)
        # x = self.drop2(x)
        # x = fun.relu(self.dnn3(x))
        # x = self.bn3(x)
        # x = self.drop3(x)
        # x = fun.relu(self.dnn3(x))
        # x = self.drop6(x)

        return x


class Decoder(nn.Module):
    def __init__(self, device) -> None:
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(kernel_size=3, stride=1, in_channels=32, out_channels=3, padding=1).to(device)
        self.pool1 = nn.MaxUnpool2d(stride=2, kernel_size=2).to(device)
        self.bn1 = nn.BatchNorm2d(32).to(device)
        self.drop1 = nn.Dropout(0.7).to(device)
        self.conv2 = nn.ConvTranspose2d(kernel_size=3, stride=1, in_channels=64, out_channels=32, padding=1).to(device)
        self.pool2 = nn.MaxUnpool2d(stride=2, kernel_size=2).to(device)
        self.bn2 = nn.BatchNorm2d(64).to(device)
        self.drop2 = nn.Dropout(0.7).to(device)
        self.conv3 = nn.ConvTranspose2d(kernel_size=3, stride=1, in_channels=128, out_channels=64, padding=1).to(device)
        self.pool3 = nn.MaxUnpool2d(stride=2, kernel_size=2).to(device)
        self.bn3 = nn.BatchNorm2d(128).to(device)
        self.drop3 = nn.Dropout(0.7).to(device)
        # self.conv4 = nn.ConvTranspose2d(kernel_size=3, stride=2, in_channels=256, out_channels=128, padding=1).to(
        #     device)
        # self.pool4 = nn.MaxUnpool2d(stride=1, kernel_size=2).to(device)
        # self.bn4 = nn.BatchNorm2d(128).to(device)
        # self.drop4 = nn.Dropout(0.7).to(device)

        self.dnn1 = nn.Linear(1024, 4096).to(device)
        self.dnn2 = nn.Linear(512, 1024).to(device)
        self.dnn3 = nn.Linear(256, 512).to(device)
        # self.dnn3 = nn.Linear(2048, 5000).to(device)
        self.drop1 = nn.Dropout(0.7).to(device)
        self.drop2 = nn.Dropout(0.7).to(device)
        self.drop3 = nn.Dropout(0.7).to(device)
        self.bn1 = nn.BatchNorm1d(12800).to(device)
        self.bn2 = nn.BatchNorm1d(8192).to(device)
        self.bn3 = nn.BatchNorm1d(4096).to(device)

    def forward(self, x):
        # x = self.drop3(x)
        # x = self.bn3(x)
        # x = self.dnn3(fun.relu(x))

        # x = self.drop2(x)
        # x = self.bn2(x)
        # x = self.dnn2(fun.relu(x))

        # x = self.bn1(x)
        x = fun.relu(self.dnn1(x))
        # x = self.drop1(x)

        # x = x.reshape(1, 64, 64)
        # x = self.pool4(x, ind4)
        # x = self.drop4(x)
        # x = self.conv4(x)
        # x = fun.relu(self.bn4(x))

        # x = self.drop3(x)
        # x = self.pool3(x, ind3)
        # x = self.bn3(fun.relu(x))
        # x = self.conv3(x)
        #
        # x = self.drop2(x)
        # x = self.pool2(x, ind2)
        # x = self.bn2(fun.relu(x))
        # x = self.conv2(x)
        #
        # x = self.drop1(x)
        # x = self.pool1(x, ind1)
        # x = self.bn1(fun.relu(x))
        # x = self.conv1(x)

        return x


class AutoCoderHelper:

    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda:0")
        self.net = AutoCoder1(self.device)