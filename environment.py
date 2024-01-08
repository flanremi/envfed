# getPCA25()
import json
import time

from enum import Enum

import torch

import train_class


class Type(Enum):
    crossing = "crossing"
    high_way = "high_way"
    main_road = "main_road"
    total = "total"
class Environment:

    def __init__(self, _type:Type):
        super().__init__()
        self.meta_url = "C:\\Users\\lily\\PycharmProjects\\zhangruoyi\\yolov5\\results2\\metadata"
        self.client_num = 25
        self._type = _type
        self.lamda = 0.5
        # 均一化系数
        self.sigma = 20
        with open(self.meta_url, "r") as file:
            self.metadata = json.loads(file.read()).get(_type)
        # 标记client是否被选中
        self.tag = [0 for i in range(self.client_num)]
        self.now_loss = 1
        self.latency = 0
        self.model = None
        self.step = 0
        self.last_reward = 0
        self.opt = train_class.Opt(
            weights='C:\\Users\\lily\\PycharmProjects\\zhangruoyi\\yolov5\\results\\{}\\client3\\train\\weights\\best.pt',
            device='0',
            data='C:\\Users\\lily\\PycharmProjects\\Finland_road_data\\yolo_data\\ymls\\{}\\val.yaml'.format(_type), epochs=1,
            )
        self.helper = train_class.TrainingHelper(self.opt)
        # 先跑一次吧数据集缓存起来
        self.helper.model_val(torch.load(
            'C:\\Users\\lily\\PycharmProjects\\zhangruoyi\\yolov5\\results\\{}\\client3\\train\\weights\\best.pt'
            .format(_type))['model'].to(torch.device("cuda")))

    def reset(self):
        self.step = 0
        self.model = None
        self.latency = 0
        self.now_loss = 1
        self.tag = [0 for i in range(self.client_num)]
        self.last_reward = 0


    def get_state(self):
        states = [self.now_loss, self.latency, ]

        for i in range(self.client_num):
            if self.tag[i] == 1:
                states.extend([0 for i in range(25)])
            else:
                states.extend(self.metadata[i]["PCA25"])
        return  states

    def get_loss(self):
        self.now_loss = self.helper.model_val(self.model)[0][4]
    def get_reward(self):
        loss = self.now_loss * self.sigma
        return 1 / (self.lamda * loss + (1 - self.lamda) * self.latency)

    # 目前定义26个维度，25个client+1个终止
    def next(self, action, decision_time):
        valid = True
        if action <= 25:
            _t = time.time()
            if self.tag[action] == 0:
                # 融合
                if self.step == 0:
                    self.model = torch.load(self.metadata[action]["model_url"])['model'].to(torch.device("cuda"))
                else:
                    self.model = train_class.sum_model(self.model,
                                                       torch.load(self.metadata[action]["model_url"])['model']
                                                       .to(torch.device("cuda")),
                                                       self.step)
                self.step += 1
                self.tag[action] = 1
                self.get_loss()
            else:
                valid = False
                # 额外的惩罚
                self.latency += 1

            self.latency += time.time() - _t + decision_time
            self.last_reward = self.get_reward()
            return self.get_state(), self.last_reward, 0, valid
        return self.get_state(), self.last_reward, 1, False
