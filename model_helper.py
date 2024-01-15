import os
import random

import torch

from train_class import getParamlistByModel
from autoencoder import AutoCoderDnn

def get_all_filenames(folder_path):
    filenames = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            filenames.append(os.path.join(root, file))
    return filenames


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
        self.net = AutoCoderDnn(self.device)
        torch.save(self.net, "tmp.pt")

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
        _t = tmp[: length]

        return


    def train(self):
        pass

ModelHelper()