# getPCA25()
import gc
import json
import time

from enum import Enum

import numpy as np
import torch
import torch.nn as nn

import os
import torch
import argparse
from tqdm import tqdm

from torchsummary import summary
from pca_helper import get_pca_by_model

from train_helper import TrainingHelper, Opt


class Type(Enum):
    crossing = "crossing"
    high_way = "high_way"
    main_road = "main_road"
    total = "total"


def get_F1(prec, recall):
    return 2 * (prec * recall) / (prec + recall)


class Environment:

    def __init__(self, _type: str):
        super().__init__()
        # self.meta_url = "C:\\Users\\lily\\PycharmProjects\\zhangruoyi\\yolov5\\results2\\metadata"
        self.device = torch.device("cuda:0")
        self.latency = [0.2987, 0.4822, 0.3846, 0.3974, 0.2972, 0.2052, 0.3582, 0.1177, 0.2522, 0.2263, ]
        self.client_num = 10
        self._type = _type
        self.lamda = 0.5
        self.delta = 1.5
        self.lvl = [(1, 240), (2, 480), (3, 640)]

        opts = [Opt(
            weights='./init_model/model{}.pt'.format(i),
            device='0',
            # epoch会实时影响学习率，因此我们应该假定A是多少轮的模型，然后每次只执行其中的若干步， 因此epoch和helper的lvl属性务必认真填
            data='C:\\Users\\lily\\PycharmProjects\\zhangruoyi\\yolov5\\yml\\client{}.yml'.format(i),
            epochs=((i % 3) + 1) * 40,
            imgsz=self.lvl[i % 3][1], ) for i in range(10)]

        self.client = [TrainingHelper(opts[i], i % 3 + 1, (i % 3 + 1) * 5) for i in
                       range(self.client_num)]

        val_opts = Opt(
            weights='C:\\Users\\lily\\PycharmProjects\\zhangruoyi\\yolov5\\yolov5n.pt',
            device='0',
            # epoch会实时影响学习率，因此我们应该假定A是多少轮的模型，然后每次只执行其中的若干步， 因此epoch和helper的lvl属性务必认真填
            data='C:\\Users\\lily\\PycharmProjects\\zhangruoyi\\yolov5\\yml\\val.yml',
            epochs=128,
            imgsz=640, )

        self.val_client = TrainingHelper(val_opts, 3, 0)
        self.epoch = 25
        self.step = 0
        # 标记阶段性成果
        self.milestone = [0 for i in range(20)]

        self.last_reward = -1
        self.f1 = 0
        self.precs = [-1 for i in range(10)]
        self.recalls = [-1 for i in range(10)]
        self.init()
        # # 初始先让所有的client推进第一步 过时
        # for client in self.client:
        #     client.train()

    def reset(self):
        opts = [Opt(
            weights='./init_model/model{}.pt'.format(i),
            device='0',
            # epoch会实时影响学习率，因此我们应该假定A是多少轮的模型，然后每次只执行其中的若干步， 因此epoch和helper的lvl属性务必认真填
            data='C:\\Users\\lily\\PycharmProjects\\zhangruoyi\\yolov5\\yml\\client{}.yml'.format(i),
            epochs=((i % 3) + 1) * 40,
            imgsz=self.lvl[i % 3][1], ) for i in range(10)]

        self.client = [TrainingHelper(opts[i], i % 3 + 1, (i % 3 + 1) * 5) for i in
                       range(self.client_num)]

        val_opts = Opt(
            weights='C:\\Users\\lily\\PycharmProjects\\zhangruoyi\\yolov5\\yolov5n.pt',
            device='0',
            # epoch会实时影响学习率，因此我们应该假定A是多少轮的模型，然后每次只执行其中的若干步， 因此epoch和helper的lvl属性务必认真填
            data='C:\\Users\\lily\\PycharmProjects\\zhangruoyi\\yolov5\\yml\\val.yml',
            epochs=128,
            imgsz=640, )

        self.val_client = TrainingHelper(val_opts, 3, 0)
        gc.collect()
        self.last_reward = -1
        self.step = 0
        self.f1 = 0
        self.milestone = [0 for i in range(20)]
        self.precs = [-1 for i in range(10)]
        self.recalls = [-1 for i in range(10)]
        self.init()

    def get_latency(self, actions):
        tmp = 0
        for action in actions:
            if self.latency[action] > tmp:
                tmp = self.latency[action]
        return tmp

    def get_state(self):
        states = []
        # state = 10位global, 10*(10local+1latency) 一共120位 / 0 + 1  + 10 * (5 + 0) = 74
        # models = [self.val_client.model.state_dict()]
        # models.extend([self.client[i].model.state_dict() for i in range(self.client_num)])
        # pca = get_pca_by_model(models)
        # states.extend(pca[0])
        states.append(self.f1)
        for i in range(1, self.client_num + 1):
            # states.extend(pca[i])
            states.append(self.precs[i - 1])
            states.append(self.recalls[i - 1])
            states.append(self.latency[i - 1])
            states.append(self.client[i - 1].epoch_now)
            states.append(self.client[i - 1].lvl)
        return states

    def get_reward(self, f1, latency):
        if self.step == self.epoch:
            now_reward = self.lamda * f1 * self.delta - (1 - self.lamda) * latency
            self.last_reward = now_reward
            return now_reward
        else:
            tmp = 0
            for i in range(int(f1 / 0.05)):
                if self.milestone[i] == 0:
                    self.milestone[i] = 1
                    tmp += 0.05
            return self.lamda * tmp * self.delta - (1 - self.lamda) * latency / 25

    # 往global中填充增量
    def sum_model(self, actions):
        # 根据train次数为参数赋权
        # 总train次数
        sum_train_times = 0
        ## tmp_locals = []
        # 训练出增量
        for action in actions:
            # tmp_locals.append(self.client[action].model.state_dict())
            self.client[action].model.load_state_dict(self.val_client.model.state_dict())
            self.client[action].train()
            sum_train_times += self.client[action].epochs
        # 带增量的模型, (模型， 权重)
        # models = [(self.client[actions[i]].model.state_dict(), self.client[actions[i]].epochs / sum_train_times) for i
        #           in range(len(actions))]
        # 带增量的模型, (模型， 权重) 使用平权
        models = [(self.client[actions[i]].model.state_dict(), 1 / 3) for i
                  in range(len(actions))]
        tmp_global = {}
        for key, var in self.val_client.model.state_dict().items():
            new = var.clone()
            # 给local附加增量
            for i in range(3):
                ## tmp_local = tmp_locals[i]
                now_local = models[i][0]
                add = now_local.get(key) - var
                # local模型迭代 local的增量 + local的原值
                ## tmp_local.update({key: add + tmp_local.get(key)})
                # global增量迭代 增量 * 权
                if new.dtype == torch.int64:
                    new += (add * models[i][1]).long()
                else:
                    new += add * models[i][1]
            # global更新 总增量 + 初值
            tmp_global.update({key: new})
        # 还原local
        ## for pos, action in enumerate(actions):
        ##     self.client[action].model.load_state_dict(tmp_locals[pos])
        self.val_client.model.load_state_dict(tmp_global)

    # 目前定义25个维度 action是從0開始的
    def next(self, actions):
        # 1、把global丢到对应车辆上训练, 过程全整合在sum_model中
        self.sum_model(actions)
        # # 临时存储client上的model
        # tmp = []
        # for action in actions:
        #     tmp.append(self.client[action].model.state_dict())
        #     self.client[action].model.load_state_dict(self.global_model)
        #     self.client[action].train()
        # self.sum_model(actions)
        # 恢复client的model
        # for pos, action in enumerate(actions):
        #     self.client[action].model.load_state_dict(tmp[pos])
        # # 2、让每个client的模型向前推进一格 过时
        # for pos in range(len(self.client)):
        #     self.client[pos].train()
        # 3、测试global
        mp, mr, map50, _, _, _, _ = self.val_client.val()
        self.f1 = get_F1(mp, mr)
        # 更新迭代过的模型的ap
        for action in actions:
            tmp = self.client[action].val()
            self.recalls[action] = tmp[1]
            self.precs[action] = tmp[0]
        # 4、获得next_state
        states = self.get_state()
        # 5、结算epoch
        self.step += 1
        if self.step >= self.epoch:
            done = 1
        else:
            done = 0
        reward = self.get_reward(self.f1, self.get_latency(actions))

        return states, reward, done, (mp, mr, map50)

    def init(self):
        # 获取所有模型的初始ap
        for i in range(10):
            tmp = self.client[i].val()
            self.recalls[i] = tmp[1]
            self.precs[i] = tmp[0]
