import math
import os
import random

import numpy
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F

current_path = os.path.dirname(__file__) + "/"

class Qnet(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''
    def __init__(self, input_dim, action_dim,):
        super(Qnet, self).__init__()
        hidden = 2 ** (int(math.log2(input_dim)))
        self.fc1 = torch.nn.Linear(input_dim, hidden)
        self.ln1 = torch.nn.LayerNorm(hidden)
        self.fc2 = torch.nn.Linear(hidden, int(hidden / 4))
        self.ln2 = torch.nn.LayerNorm(int(hidden / 4))
        self.fc3 = torch.nn.Linear(int(hidden / 4), int(hidden / 4))
        self.ln3 = torch.nn.LayerNorm(int(hidden / 4))

        self.fc4 = torch.nn.Linear(int(hidden / 4), action_dim)

    def forward(self, x):
        x = self.ln1(F.relu(self.fc1(x)))  # 隐藏层使用ReLU激活函数
        x = self.ln2(F.relu(self.fc2(x)))  # 隐藏层使用ReLU激活函数
        x = self.ln3(F.relu(self.fc3(x)))
        return self.fc4(x)


class DDQN:
    ''' DQN算法 '''

    def __init__(self, device, input_dim, action_dim, name, sigma = 0.01, actor_lr = 3e-4, learning_rate = 0.001,
                 gamma = 0.95, tau = 0.005, epsilon = 0.05, target_update = 10):
        self.action_dim = action_dim
        self.input_dim = input_dim
        self.name = name
        self.q_net = Qnet(self.input_dim, self.action_dim).to(device)  # Q网络
        # 目标网络
        self.target_q_net = Qnet(self.input_dim, self.action_dim).to(device)
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate)
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device

    def take_action(self, state):  # epsilon-贪婪策略采取动作
        if np.random.random() < self.laos:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def save_net(self):
        global current_path
        torch.save(self.q_net.state_dict(), current_path + "net/" + self.name +  "_dqn_q_net")
        torch.save(self.target_q_net.state_dict(), current_path + "net/"+ self.name +  "_dqn_target_q_net")

    def load_net(self):
        global current_path
        self.q_net.load_state_dict(torch.load(current_path + "net/" + self.name+ "_dqn_q_net"))
        self.target_q_net.load_state_dict(torch.load(current_path + "net/"+ self.name + "_dqn_target_q_net"))

    # Q是向后传递的，期望是向前迭代的
    def update(self, transition_dict):
        states = torch.tensor(transition_dict[0],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict[1]).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict[2],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict[3],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict[4],
                             dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)  # Q值

        # ddqn的不同 dqn是直接max取出target中的Qvalue, 而DDQN的决策是由q_net做出的，由q_net选择target得出的向量中的值
        # actiont = self.q_net(next_states)
        # action2 = actiont.argmax(dim=1).unsqueeze(dim=1)
        # max_next_q_values = self.target_q_net(next_states).gather(1, action2)
        # 仅用DQN
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(
            -1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones
                                                                )  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        # print(dqn_loss)
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # 更新目标网络
        self.count += 1

        return dqn_loss.detach().cpu().item()
