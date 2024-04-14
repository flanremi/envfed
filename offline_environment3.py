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
import threading


class Type(Enum):
    crossing = "crossing"
    high_way = "high_way"
    main_road = "main_road"
    total = "total"


def get_F1(prec, recall):
    return 2 * (prec * recall) / (prec + recall)


def load_model(epoch, container, check):
    for j in range(10):
        container[j][epoch] = \
            torch.load('C:\\Users\\lily\\PycharmProjects\\zhangruoyi\\yolov5\\base_models\\client{}\\epoch{}.pt'
                       .format(j, epoch))['model'].state_dict()
        print(str(j) + "======" + str(epoch))
    check[epoch] = True


def load_model2(epoch, container, check, _type):
    for j in range(10):
        container[j][epoch] = \
            torch.load('C:\\Users\\lily\\PycharmProjects\\zhangruoyi\\yolov5\\base_models4\\{}\\client{}\\epoch{}.pt'
                       .format(_type, j, epoch))['model'].cpu().state_dict()
        print(str(j) + "======" + str(epoch))
    check[epoch] = True


# 简化版环境，prec,recall直接取选择的模型的平均值
class Environment:

    def __init__(self, _type: str):
        super().__init__()
        # self.meta_url = "C:\\Users\\lily\\PycharmProjects\\zhangruoyi\\yolov5\\results2\\metadata"
        self.device = torch.device("cuda:0")
        # self.latency = [0.2987, 0.4822, 0.3846, 0.3974, 0.2972, 0.2052, 0.3582, 0.1177, 0.2522, 0.2263, ]
        self.client_num = 10
        self._type = _type
        self.lamda = 0.5
        self.delta = 1.5
        self.lvl = [(1, 240), (2, 480), (3, 640)]
        self.epochs = [random.randint(0, 50) for i in range(10)]
        self.val_result = None

        with open("./base_models/val") as file:
            self.val_result = json.loads(file.read())
        print("载入模型" + str(time.time()))
        self.models = [[None for j in range(128)] for i in range(10)]
        # self.models = [
        #     [torch.load('C:\\Users\\lily\\PycharmProjects\\zhangruoyi\\yolov5\\base_models\\client{}\\epoch{}.pt'
        #                 .format(i, j))['model'].state_dict()
        #      for j in range(1)] for i in range(10)]
        check = [False for i in range(128)]
        for i in range(128):
            threading.Thread(target=load_model, args=(i, self.models, check)).start()
        while True:
            stop = True
            for i in range(128):
                if not check[i]:
                    stop = False
            if stop:
                break
        print("模型载入成功" + str(time.time()))

        self.client = [self.models[i][self.epochs[i]] for i in range(10)]

        # self.val_client = \
        #     torch.load('C:\\Users\\lily\\PycharmProjects\\zhangruoyi\\yolov5\\base_models\\client{}\\epoch{}.pt'
        #                .format(random.randint(0, 9), random.randint(0, 40)))['model'].state_dict()
        self.epoch = 25
        self.step = 0
        self.prec = 0
        self.recall = 0
        # 标记阶段性成果
        self.milestone = [0 for i in range(20)]

        self.last_reward = -1
        self.f1 = 0

    def reset(self):
        self.epochs = [random.randint(0, 50) for i in range(10)]
        # self.client = [
        #     torch.load('C:\\Users\\lily\\PycharmProjects\\zhangruoyi\\yolov5\\base_models\\client{}\\epoch{}.pt'
        #                .format(i, self.epochs[i]))['model'].state_dict() for i in range(10)]
        self.client = [self.models[i][self.epochs[i]] for i in range(10)]
        # self.val_client = \
        #     torch.load('C:\\Users\\lily\\PycharmProjects\\zhangruoyi\\yolov5\\base_models\\client{}\\epoch{}.pt'
        #                .format(random.randint(0, 9), random.randint(0, 40)))['model'].state_dict()
        self.last_reward = -1
        self.step = 0
        self.f1 = 0
        self.prec = 0
        self.recall = 0
        self.milestone = [0 for i in range(20)]

    def get_latency(self, actions):
        tmp = 0
        for action in actions:
            if self.latency[action] > tmp:
                tmp = self.latency[action]
        return tmp

    def get_state(self):
        states = []
        # state = 10位global, 10*(10local+1latency) 一共120位 /  0  + 10 * (4 + 1) = 54
        # models = [self.val_client]
        # models = []
        # models.extend(self.client)
        pca = get_pca_by_model(self.client)
        # states.extend(pca[0])
        for i in range(1, self.client_num + 1):
            states.extend(pca[i - 1])
            # states.append(self.precs[i - 1])
            # states.append(self.recalls[i - 1])
            # states.append(self.latency[i - 1])
            # states.append(self.client[i - 1].epoch_now)
            # states.append(self.client[i - 1].lvl)
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

    def sum_model(self, actions):
        models = [self.client[actions[i]] for i
                  in range(len(actions))]
        tmp_global = {}
        for key, var in self.val_client.items():
            # 直接叠加
            tmp_global.update({key: (models[0].get(key) + models[1].get(key) + models[2].get(key)) / 3})
        return tmp_global

    # 目前定义25个维度 action是從0開始的
    def next(self, actions):
        # 更换选定action的模型,并计算prec和re
        prec = 0
        re = 0
        for action in actions:
            prec += self.val_result[action][self.epochs[action]][0]
            re += self.val_result[action][self.epochs[action]][1]
            self.epochs[action] += self.lvl[action % 3][0]
            # self.client[action] = (
            #     torch.load('C:\\Users\\lily\\PycharmProjects\\zhangruoyi\\yolov5\\base_models\\client{}\\epoch{}.pt'
            #                .format(action, self.epochs[action]))['model'].state_dict())
            self.client[action] = self.models[action][self.epochs[action]]
        # 更换val模型
        # self.val_client = self.sum_model(actions)
        self.prec = (prec + self.prec) / 4
        self.recall = (re + self.recall) / 4
        # 计算f1
        self.f1 = get_F1(self.prec, self.recall)
        # 4、获得next_state
        states = self.get_state()
        # 5、结算epoch
        self.step += 1
        if self.step >= self.epoch:
            done = 1
        else:
            done = 0
        reward = self.get_reward(self.f1, self.get_latency(actions))

        return states, reward, done, (self.prec, self.recall, self.f1)


# if __name__ == '__main__':
#     aa = torch.load("./init_model/model0.pt")
#     bb = aa['model'].state_dict()
#     print(1)
if __name__ == '__main__':
    # 生产PCA信息
    for _type in ["main_road"]:
        # 存储20000条
        models = [[None for j in range(150)] for i in range(10)]
        check = [False for i in range(150)]
        for i in range(150):
            threading.Thread(target=load_model2, args=(i, models, check, _type)).start()
        while True:
            stop = True
            for i in range(150):
                if not check[i]:
                    stop = False
            if stop:
                break
        print("模型载入成功" + str(time.time()))
        containers = []
        for pos in range(20000):
            # 1、 随机取10个model
            epochs = [random.randint(0, 149) for i in range(10)]
            # 2、 得到pca
            tmp = [models[i][epochs[i]] for i in range(10)]
            pca = get_pca_by_model(tmp)
            # 3、 存储 (每个client的epoch，pca)
            containers.append((epochs, pca))
            # 4、 保存
            if pos % 100 == 99:
                print(_type + "=======" + str(pos))
                with open("./pca/" + _type + "5", "w") as file:
                    file.write(json.dumps(containers))
