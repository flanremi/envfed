import os
import random

import cv2
import torch
import numpy as np

import autoencoder
import train_class
from train_class import getParamlistByModel
from autoencoder import AutoCoderDnn

from PIL import Image

def get_all_filenames(folder_path):
    filenames = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            filenames.append(os.path.join(root, file))
    return filenames

def resize(image, x, y):
    image = Image.fromarray(image)
    out_image = image.resize((int(x), int(y)))
    return  np.array(out_image)

class ModelHelper:

    def __init__(self):
        super().__init__()
        self.dir_names = ["results", "results2"]
        self.env_names = ["crossing", "high_way", "main_road", "total"]
        self.model_dir = "C:\\Users\\lily\\PycharmProjects\\zhangruoyi\\yolov5\\{}\\{}\\{}\\train\\weights\\"
        self.pool = []
        self.val_pool = []
        self.train_pool = []
        self.init_pool()
        min = int(len(self.pool) / 15)
        max = int(len(self.pool) / 10)
        val_len = random.randint(min, max)
        tmps = self.pool.copy()
        for i in range(val_len):
            pos = random.randint(0, len(tmps) - 1)
            tmp = tmps.pop(pos)
            self.val_pool.append(tmp)
        self.train_pool.extend(tmps)
        self.device = torch.device("cuda:0")
        self.net = autoencoder.AutoCoder1(self.device)
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)
        self.lossFunc = torch.nn.MSELoss()

    def init_pool(self):
        for dir_name in self.dir_names:
            for env_name in self.env_names:
                for i in range(25):
                    url = self.model_dir.format(dir_name, env_name, "client" + str(i))
                    file_names = get_all_filenames(url)
                    j = 0
                    while j < len(file_names):
                        if file_names[j].find("epoch") == -1:
                            file_names.pop(j)
                        else:
                            j += 1
                    self.pool.extend(file_names)

    def verify_dimension(self):
        values = []
        for _item in self.pool:
            a = getParamlistByModel(_item)
            record = True
            for _v in values:
                if _v == len(a):
                    record = False
                    break
            if record:
                values.append(len(a))
                print(values)

    def getSquad(self, length):
        tmp = self.pool.copy()
        random.shuffle(tmp)
        _ts = tmp[: length]
        result = []
        for _t in _ts:
            item = train_class.getParamlistByModel(_t)
            item = np.resize(item, (1, 7031250))
            result.append(item)
        return result


    def train(self):
        pass

from torch import optim

if __name__ == '__main__':
    helper = ModelHelper()
    result = []
    # 周期
    for i in range(1000):
        a = np.array(helper.getSquad(2048))
        loss = 0
        for j in range(int(2048 / 16)):
            b = torch.from_numpy(a[j * 16: (j + 1) * 16]).to(helper.device, torch.float)
            target = torch.from_numpy(a[j * 16: (j + 1) * 16]).to(helper.device, torch.float)
            out = helper.net.forward(b)
            _loss = helper.lossFunc(out, target)
            loss += _loss.to(torch.device("cpu")).detach().numpy().reshape(-1)
            helper.optimizer.zero_grad()
            _loss.backward()
            helper.optimizer.step()
        loss /= 2048
        result.append(loss)
        print(loss)
    torch.save(helper.net.state_dict(), "auto_encoder\\model1.pt")
