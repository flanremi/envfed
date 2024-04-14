# getPCA25()
import gc
import json
import random
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
        self.client_num = 10
        self._type = _type
        # self.lvl = [(1, 240), (1, 240), (2, 240), (2, 480), (3, 640), (1, 480), (2, 240), (2, 240), (1, 240), (1, 240)]
        self.lvl = [(2, 640) for i in range(10)]
        img_num = {"crossing": [250, 250, 250, 41, 46, 44, 47, 49, 49, 47],
                   "main_road": [272, 272, 272, 29, 36, 33, 45, 43, 38, 34],
                   "high_way": [243, 243, 243, 29, 48, 37, 41, 32, 44, 40]}
        self.img_num = img_num.get(_type).copy()
        self.latency = [0.2987, 0.4822, 0.3846, 0.3974, 0.2972, 0.2052, 0.3582, 0.1177, 0.2522, 0.2263]
        self.latency_add = [random.randint(-1000, 1000) / 1000 * 0.05 for i in range(10)]

        opts = [Opt(
            weights='C:\\Users\\lily\\PycharmProjects\\zhangruoyi\\yolov5\\base_models2\\{}\\client{}\\epoch{}.pt'.format(
                self._type, i, self.lvl[i][0] - 1),
            device='0',
            # epoch会实时影响学习率，因此我们应该假定A是多少轮的模型，然后每次只执行其中的若干步， 因此epoch和helper的lvl属性务必认真填
            data='C:\\Users\\lily\\PycharmProjects\\zhangruoyi\\yolov5\\yml\\{}\\client{}.yml'.format(self._type, i),
            epochs=150,
            imgsz=self.lvl[i][1], ) for i in range(10)]

        self.client = [TrainingHelper(opts[i], self.lvl[i][0], 0) for i in
                       range(self.client_num)]

        val_opts = Opt(
            weights='C:\\Users\\lily\\PycharmProjects\\zhangruoyi\\yolov5\\yolov5n.pt',
            device='0',
            # epoch会实时影响学习率，因此我们应该假定A是多少轮的模型，然后每次只执行其中的若干步， 因此epoch和helper的lvl属性务必认真填
            data='C:\\Users\\lily\\PycharmProjects\\zhangruoyi\\yolov5\\yml\\{}\\val.yml'.format(self._type),
            epochs=150,
            imgsz=640, )

        self.val_client = TrainingHelper(val_opts, 3, 0)
        self.epoch = 75
        self.step = 0


    def reset(self):
        opts = [Opt(
            weights='C:\\Users\\lily\\PycharmProjects\\zhangruoyi\\yolov5\\base_models2\\{}\\client{}\\epoch{}.pt'.format(
                self._type, i, self.lvl[i][0] - 1),
            device='0',
            # epoch会实时影响学习率，因此我们应该假定A是多少轮的模型，然后每次只执行其中的若干步， 因此epoch和helper的lvl属性务必认真填
            data='C:\\Users\\lily\\PycharmProjects\\zhangruoyi\\yolov5\\yml\\{}\\client{}.yml'.format(self._type, i),
            epochs=150,
            imgsz=self.lvl[i][1], ) for i in range(10)]

        self.client = [TrainingHelper(opts[i], self.lvl[i][0], 0) for i in
                       range(self.client_num)]

        val_opts = Opt(
            weights='C:\\Users\\lily\\PycharmProjects\\zhangruoyi\\yolov5\\yolov5n.pt',
            device='0',
            # epoch会实时影响学习率，因此我们应该假定A是多少轮的模型，然后每次只执行其中的若干步， 因此epoch和helper的lvl属性务必认真填
            data='C:\\Users\\lily\\PycharmProjects\\zhangruoyi\\yolov5\\yml\\{}\\val.yml'.format(self._type),
            epochs=150,
            imgsz=640, )

        self.val_client = TrainingHelper(val_opts, 3, 0)
        gc.collect()

        self.step = 0
        self.latency_add = [random.randint(-1000, 1000) / 1000 * 0.05 for i in range(10)]


    def get_state(self):
        states = []
        models = []
        models.extend([self.client[i].model.state_dict() for i in range(self.client_num)])
        pca = get_pca_by_model(models)
        for i in range(0, self.client_num):
            states.extend(pca[i])
        return states

    # 往global中填充增量
    def sum_model(self, actions):
        # 根据train次数为参数赋权
        # 总train次数
        tmp_locals = []
        total_img_num = 0
        # 训练出增量
        for action in actions:
            total_img_num += self.img_num[action]
            tmp_locals.append(self.client[action].model.state_dict())
            self.client[action].model.load_state_dict(self.val_client.model.state_dict())
            self.client[action].train()
        # 带增量的模型, (模型， 权重) 使用平权
        models = [(self.client[actions[i]].model.state_dict(), self.img_num[actions[i]] / total_img_num) for i
                  in range(len(actions))]
        tmp_global = {}
        for key, var in self.val_client.model.state_dict().items():
            new = var.clone()
            # 给local附加增量
            for i in range(len(actions)):
                tmp_local = tmp_locals[i]
                now_local = models[i][0]
                add = now_local.get(key) - var
                # local模型迭代 local的增量 + local的原值
                tmp_local.update({key: add + tmp_local.get(key)})
                # global增量迭代 增量 * 权
                if new.dtype == torch.int64:
                    new += (add * models[i][1]).long()
                else:
                    new += add * models[i][1]
            # global更新 总增量 + 初值
            tmp_global.update({key: new})
        # 还原local
        for pos, action in enumerate(actions):
            self.client[action].model.load_state_dict(tmp_locals[pos])
        self.val_client.model.load_state_dict(tmp_global)

    # 目前定义25个维度 action是從0開始的
    def next(self, actions):
        self.sum_model(actions)
        self.step += 1
        # 3、测试global
        mp, mr, map50, _, _, _, _ = self.val_client.val()
        f1 = get_F1(mp, mr) -
        done = 0
        if self.step >= self.epoch:
            done = 1
        self.latency_add = [random.randint(-1000, 1000) / 1000 * 0.05 for i in range(10)]
        return f1, (mp, mr, map50), done, [self.latency[i] + self.latency_add[i] for i in range(10)]
