import json
import math
import random
import time

import numpy as np
import torch

from dqn.state_container import CacheContainer

import dqn.latency_dqn_agent as net
import dqn.latency_dual_nstep_noisy_dqn_agent2 as net2

from dqn.replay_buffer import ReplayBuffer, ReplayBufferN
import latency_new_environment

num_episodes = 2000
buffer_size = 400000

minimal_size = 64
batch_size = 64
update_interval = 16
# 1， 3， 5
n_steps = [1, 3, 5]


def get_random_action():
    a = [i for i in range(10)]
    random.shuffle(a)
    # return a[0:3]
    return [1, 2, 3]


def get_reward(val, epochs, actions, latency, lamda, _type):
    area = {'main_road': [1, 1.1357142857142857, 1.2252176250052065, 1.2932699182663003, 1.3481909658939082, 1.394237840444128,
      1.4338833651959593, 1.468692051567313, 1.4997168103508136, 1.5277002473145453],
            'high_way': [1, 1.1833333333333333, 1.2877538958394075, 1.3671482379773503, 1.431222793542893, 1.484944147184816,
      1.5311972593952858, 1.5718073934951984, 1.6080029454092826, 1.6406502885336363],
            'crossing': [1, 1.153030303030303, 1.2479580871267342, 1.3201347617975911, 1.3783843577662664, 1.4272219519861964,
      1.4692702358138963, 1.5061885395410894, 1.5390935867357114, 1.568772989576033]}
    prec = 0
    recall = 0
    ap = 0
    for action in actions:
        prec += val[action][epochs[action]][0]
        if val[action][epochs[action]][1] > recall:
            recall = val[action][epochs[action]][1]
        ap += val[action][epochs[action]][2]
    prec /= len(actions)
    # recall /= len(actions)
    recall *= area[_type][len(actions) - 1]
    # ap /= len(actions)
    # F1
    tmp = 1 * lamda[0] * ((1 + 1 ** 2) * prec * recall / ((1 ** 2) * prec + recall + 0.00000001)) - lamda[1] * latency
    return tmp



# [(1, 240), (1, 240), (2, 240), (2, 480), (3, 640), (1, 480), (2, 240), (2, 240), (1, 240), (1, 240)]
def prefer_p(_type):
    actions = [3, 4, 5]
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    env = latency_new_environment.Environment(_type)
    result = []
    for i in range(75):
        f1, (mp, mr, map50), done = env.next(actions)
        result.append((f1, (mp, mr, map50)))
    with open("./dqn_result/p/" + _type + "2", "w") as file:
        file.write(json.dumps(result))


def get_actions(latencys, latency_adds, action):
    latency = latencys[action] + latency_adds[action]
    results = []
    for i in range(10):
        if latencys[i] + latency_adds[i] <= latency:
            results.append(i)
    return results


# 直接生产经验
if __name__ == '__main__':
    for _type in ['main_road', 'high_way', 'crossing']:
        for la in [(0.1, 0.9), (0.3, 0.7), (0.5, 0.5), (0.7, 0.3), (0.9, 0.1)]:
            # prefer_p(_type)
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            with open("./pca/" + _type + "5", "r") as file:
                pcas = json.loads(file.read())
            with open("./base_models4/" + _type + "/val", "r") as file:
                val = json.loads(file.read())

            replay_bufferN = ReplayBufferN(buffer_size, batch_size)
            agent = net.DDQN(device, 2 + 10 + 10 * (4), 10, name=_type + "/latency_5_{}_".format(la[0]), epsilon=0)
            # agent.load_net()
            agent2 = net2.DDQN(device, 2 + 10 + 10 * (4), 10, name=_type + "/latency_5_{}_".format(la[0]), epsilon=0)
            # agent2.load_net()
            dqn_loss = []
            ndqn_loss = []
            # print(1)
            # 每条生成10条经验
            latency = [0.1177, 0.2052, 0.2263, 0.3082, 0.2572, 0.2987, 0.4022, 0.3346, 0.3474, 0.2522]
            for pca in pcas:
                state = []
                for pc in pca[1]:
                    state.extend(pc)
                for i in range(10):
                    _state = state.copy()
                    latency_add = [random.randint(-1000, 1000) / 1000 * 0.05 for i in range(10)]
                    action = i % 10
                    _state.extend(la)
                    _state.extend([latency[i] + latency_add[i] for i in range(10)])
                    latenc = latency[action] + latency_add[action]
                    actions = get_actions(latency, latency_add, action)
                    # replay_buffer.add(pca[1], actions, get_reward(val, pca[0], actions), [0 for i in range(40)], 1)
                    replay_bufferN.add(_state, action, get_reward(val, pca[0], actions, latenc, la, _type), [0 for i in range(52)], 1, 1)
            print("以生产完经验")
            for i in range(num_episodes):
                sample = replay_bufferN.sample()
                dqn_loss.append(agent.update(sample))
                if i % 100 == 99:
                    agent.save_net()
                    print("agent====" + str(i))
            with open("./dqn/net/" + _type + "/loss_dqn_l5_" + str(la[0]), "w") as file:
                file.write(json.dumps(dqn_loss))
            for i in range(num_episodes):
                sample = replay_bufferN.sample()
                ndqn_loss.append(agent2.update(sample))
                if i % 100 == 99:
                    agent2.save_net()
                    print("agent2====" + str(i))
            with open("./dqn/net/" + _type + "/loss_ndqn_l5" + str(la[0]), "w") as file:
                file.write(json.dumps(ndqn_loss))
        # # 每条生成100条经验
        # for pca in pcas:
        #     state = []
        #     for pc in pca[1]:
        #         state.extend(pc)
        #     tmp = [i for i in range(10)]
        #     for i in range(100):
        #         random.shuffle(tmp)
        #         actions = tmp[:7]
        #         # replay_buffer.add(pca[1], actions, get_reward(val, pca[0], actions), [0 for i in range(40)], 1)
        #         replay_bufferN.add(state, actions, get_reward(val, pca[0], actions), [0 for i in range(40)], 1, 1)
        # print("以生产完经验")
        # for i in range(num_episodes):
        #     sample = replay_bufferN.sample()
        #     dqn_loss.append(agent.update(sample))
        #     if i % 100 == 99:
        #         agent.save_net()
        #         print("agent====" + str(i))
        # with open("./dqn/net/" + _type + "/loss_dqn7", "w") as file:
        #     file.write(json.dumps(dqn_loss))
        # for i in range(num_episodes):
        #     sample = replay_bufferN.sample()
        #     ndqn_loss.append(agent2.update(sample))
        #     if i % 100 == 99:
        #         agent2.save_net()
        #         print("agent2====" + str(i))
        # with open("./dqn/net/" + _type + "/loss_ndqn7", "w") as file:
        #     file.write(json.dumps(ndqn_loss))
