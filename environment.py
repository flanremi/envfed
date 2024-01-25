# getPCA25()
import json
import time

from enum import Enum

import numpy as np
import torch
import torch.nn as nn

import autoencoder
import train_class
from autoencoder import AutoCoder1


class Type(Enum):
    crossing = "crossing"
    high_way = "high_way"
    main_road = "main_road"
    total = "total"
class Environment:

    def __init__(self, _type:str):
        super().__init__()
        # self.meta_url = "C:\\Users\\lily\\PycharmProjects\\zhangruoyi\\yolov5\\results2\\metadata"
        self.device = torch.device("cuda:0")
        self.net = nn.DataParallel(autoencoder.AutoCoder1(self.device)).to(self.device)
        self.net.load_state_dict(torch.load("C:\\Users\\lily\\PycharmProjects\\zhangruoyi\\yolov5\\auto_encoder\\model1.pt"))

        self.client_num = 25
        self._type = _type
        self.lamda = 0.5
        # 均一化系数
        self.sigma = 40
        # with open(self.meta_url, "r") as file:
        #     self.metadata = json.loads(file.read()).get(_type)
        with open("C:\\Users\\lily\\PycharmProjects\\zhangruoyi\\yolov5\\auto_encoder\\codes", "r") as file:
            self.codes = json.loads(file.read())[_type]
        self.model_urls = ["C:\\Users\\lily\\PycharmProjects\\zhangruoyi\\yolov5\\results2\\{}\\client{}\\train\\weights\\last.pt"
                           .format(_type, i)
                           for i in range(self.client_num)]
        # 标记client是否被选中
        self.tag = [0 for i in range(self.client_num)]
        self.now_loss = 1
        self.latency = 0
        self.model = None
        self.step = 0
        self.last_reward = -1
        # opt不涉及到模型，所以可以隨便弄
        self.opt = train_class.Opt(
            weights='C:\\Users\\lily\\PycharmProjects\\zhangruoyi\\yolov5\\results2\\{}\\client0\\train\\weights\\best.pt',
            device='0',
            data='C:\\Users\\lily\\PycharmProjects\\Finland_road_data\\yolo_data\\ymls\\{}\\val.yaml'.format(_type), epochs=1,
            )
        self.helper = train_class.TrainingHelper(self.opt)
        # 先跑一次吧数据集缓存起来
        self.helper.model_val(torch.load(
            'C:\\Users\\lily\\PycharmProjects\\zhangruoyi\\yolov5\\results2\\{}\\client0\\train\\weights\\best.pt'
            .format(_type))['model'].to(torch.device("cuda")))


    def reset(self):
        self.step = 0
        self.model = None
        self.latency = 0
        self.now_loss = 1
        self.tag = [0 for i in range(self.client_num)]
        self.last_reward = -1


    def get_state(self):
        states = [self.now_loss, self.latency, ]
        # 添加當前模型的向量值
        if self.model is None:
            states.extend([0 for i in range(16)])
        else:
            item = train_class.expandModel(self.model)
            item = np.resize(item, (1, 7031250))
            x = torch.from_numpy(np.array([item])).to(self.device, torch.float)
            now_model = self.net.forward(x).detach().cpu().numpy()
            now_model = np.resize(now_model, (16)).tolist()
            states.extend(now_model)
        # 添加其他模型的向量值
        for i in range(self.client_num):
            if self.tag[i] == 1:
                states.extend([0 for i in range(16)])
            else:
                states.extend(self.codes[i])
        return  states

    def get_loss(self):
        self.now_loss = self.helper.model_val(self.model)[0][4]
    def get_reward(self):
        loss = self.now_loss * self.sigma
        return 1 / (self.lamda * loss + (1 - self.lamda) * self.latency)

    # 目前定义26个维度，25个client+1个终止, action是從0開始的
    def next(self, action, decision_time):
        valid = True
        if action < 25:
            _t = time.time()
            if self.tag[action] == 0:
                # 融合
                if self.step == 0:
                    self.model = torch.load(self.model_urls[action])['model'].to(torch.device("cuda"))
                else:
                    self.model = train_class.sum_model(self.model,
                                                       torch.load(self.model_urls[action])['model']
                                                       .to(torch.device("cuda")),
                                                       self.step)
                self.latency += time.time() - _t + decision_time
                self.step += 1
                self.tag[action] = 1
                self.get_loss()
            else:
                valid = False
                # 额外的惩罚
                # self.latency += 1
                self.latency += time.time() - _t + decision_time
            self.last_reward = self.get_reward()
            return self.get_state(), (self.lamda - 1) / 10 if valid else (self.lamda - 1) / 2, 0, valid
        return self.get_state(), self.last_reward, 1, False

# 生成环境模型在特征向量
# if __name__ == '__main__':
#     from train_class import getParamlistByModel
#     import torch.nn as nn
#     road_type = ["crossing", "high_way", "main_road", "total"]
#     model_url = "C:\\Users\\lily\\PycharmProjects\\zhangruoyi\\yolov5\\results2\\{}\\client{}\\train\\weights\\last.pt"
#     device = torch.device("cuda:0")
#     net = nn.DataParallel(autoencoder.AutoCoder1(device)).to(device)
#     aa = torch.load("C:\\Users\\lily\\PycharmProjects\\zhangruoyi\\yolov5\\auto_encoder\\model1.pt")
#     net.load_state_dict(aa)
#     encoder_code = {}
#     for _t in road_type:
#         tmps = []
#         for i in range(25):
#             url = model_url.format(_t, i)
#             item = train_class.getParamlistByModel(url)
#             item = np.resize(item, (1, 7031250))
#             x = torch.from_numpy(np.array([item])).to(device, torch.float)
#             x = net.forward(x)
#             x = x.detach().cpu().numpy()
#             x = np.resize(x, (16)).tolist()
#             tmps.append(x)
#         encoder_code.update({_t: tmps})
#     with open("C:\\Users\\lily\\PycharmProjects\\zhangruoyi\\yolov5\\auto_encoder\\codes", "w") as file:
#         file.write(json.dumps(encoder_code))
