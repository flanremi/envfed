import json
import random
import time

import numpy as np
import torch

from dqn.state_container import CacheContainer

import dqn.now_dqn_agent as net
import dqn.now_dual_nstep_noisy_dqn_agent2 as net2

from dqn.replay_buffer import ReplayBuffer, ReplayBufferN
import new_environment

num_episodes = 5000
buffer_size = 2000000

minimal_size = 64
batch_size = 512
update_interval = 16
# 1， 3， 5
n_steps = [1, 3, 5]


def get_random_action():
    a = [i for i in range(10)]
    random.shuffle(a)
    # return a[0:3]
    return [1, 2, 3]


def get_reward(val, epochs, actions):
    prec = 0
    recall = 0
    for action in actions:
        prec += val[action][epochs[action]][0]
        recall += val[action][epochs[action]][1]
    prec /= 3
    recall /= 3
    return 2 * prec * recall / (prec + recall)


# [(1, 240), (1, 240), (2, 240), (2, 480), (3, 640), (1, 480), (2, 240), (2, 240), (1, 240), (1, 240)]
def prefer_p(_type):
    actions = [3, 4, 5]
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    env = new_environment.Environment(_type)
    result = []
    for i in range(75):
        f1, (mp, mr, map50), done = env.next(actions)
        result.append((f1, (mp, mr, map50)))
    with open("./dqn_result/p/3" + _type, "w") as file:
        file.write(json.dumps(result))


# 直接生产经验
if __name__ == '__main__':

    for _type in ['main_road', 'high_way', 'crossing']:
        prefer_p(_type)
        # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # with open("./pca/" + _type + "3", "r") as file:
        #     pcas = json.loads(file.read())
        # with open("./base_models2/" + _type + "/val", "r") as file:
        #     val = json.loads(file.read())
        # with open("./base_models2/" + _type + "/val_3", "r") as file:
        #     val3 = json.loads(file.read())
        # val[3] = val3[0]
        # val[4] = val3[1]
        # val[5] = val3[2]
        # replay_bufferN = ReplayBufferN(buffer_size, batch_size)
        # agent = net.DDQN(device, 0 + 10 * (4), 10, name=_type + "/3", epsilon=0)
        # # agent.load_net()
        # agent2 = net2.DDQN(device, 0 + 10 * (4), 10, name=_type + "/3", epsilon=0)
        # # agent2.load_net()
        # dqn_loss = []
        # ndqn_loss = []
        # # print(1)
        # # 每条生成100条经验
        # for pca in pcas:
        #     state = []
        #     for pc in pca[1]:
        #         state.extend(pc)
        #     tmp = [i for i in range(10)]
        #     for i in range(100):
        #         random.shuffle(tmp)
        #         actions = tmp[:3]
        #         # replay_buffer.add(pca[1], actions, get_reward(val, pca[0], actions), [0 for i in range(40)], 1)
        #         replay_bufferN.add(state, actions, get_reward(val, pca[0], actions), [0 for i in range(40)], 1, 1)
        # print("以生产完经验")
        # for i in range(num_episodes):
        #     sample = replay_bufferN.sample()
        #     dqn_loss.append(agent.update(sample))
        #     if i % 100 == 99:
        #         agent.save_net()
        #         print("agent====" + str(i))
        # with open("./dqn/net/" + _type + "/loss_dqn3", "w") as file:
        #     file.write(json.dumps(dqn_loss))
        # for i in range(num_episodes):
        #     sample = replay_bufferN.sample()
        #     ndqn_loss.append(agent2.update(sample))
        #     if i % 100 == 99:
        #         agent2.save_net()
        #         print("agent2====" + str(i))
        # with open("./dqn/net/" + _type + "/loss_ndqn3", "w") as file:
        #     file.write(json.dumps(ndqn_loss))
