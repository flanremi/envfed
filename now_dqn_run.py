import json
import random
import time

import numpy as np
import torch

from dqn.state_container import CacheContainer

import dqn.now_dqn_agent as net
import dqn.now_dual_nstep_noisy_dqn_agent2 as net2

from dqn.replay_buffer import ReplayBuffer, ReplayBufferN
import n0w_environment

num_episodes = 500
buffer_size = 100000

minimal_size = 64
batch_size = 32
update_interval = 16
# 1， 3， 5
n_steps = [1, 3, 5]


def get_random_action():
    a = [i for i in range(10)]
    random.shuffle(a)
    # return a[0:3]
    return [1, 2, 3]


# if __name__ == '__main__':
#     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#     agent2 = net2.DDQN(device, 0 + 1 + 10 * (0 + 3), 10, name="", epsilon=0.05)
#     agent2.load_net()
#     env = now_environment.Environment(now_environment.Type.high_way.value)
#     tmp3 = []
#     for i in range(10):
#         while True:
#             action = get_random_action()
#             next_state, reward, done, val = env.next(action)
#             tmp3.append((action, val))
#             if done == 1:
#                 break
#         env.reset()
#         with open("tmp3", "w") as file:
#             file.write(json.dumps(tmp3))


if __name__ == '__main__':
    for _type in [0, 1]:
        for area in ['main_road', "crossing", "high_way"]:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            total_step = 0
            env = n0w_environment.Environment(area)

            replay_buffer = ReplayBuffer(buffer_size, batch_size)
            replay_bufferN = ReplayBufferN(buffer_size, batch_size)
            agent = net.DDQN(device, 10 * (4 + 0), 10, name=area + "/real_", epsilon=0.05)
            agent2 = net2.DDQN(device, 10 * (4 + 0), 10, name=area + "/real_", epsilon=0.05)

            return_list = []
            loss_data = []
            # latency_data =[]
            reward_data = []
            # steps = []
            cache = []

            n_step_reward_cache = []
            n_step_state_cache = []
            n_step_counter = n_steps[_type - 1]

            for i_episode in range(num_episodes):
                # 转换下agent的位置，避免序号对训练的影响
                # ep_returns = np.zeros(len(env.agents))
                while True:
                    state = env.get_state()
                    if _type == 0:
                        action = agent.take_action(state)
                        next_state, reward, done = env.next(action)
                        replay_buffer.add(state, action, reward, next_state, done)
                        cache.append((state, action, reward, next_state, done))
                        total_step += 1
                        if replay_buffer.size(
                        ) >= minimal_size and total_step % int(update_interval) == 0:
                            sample = replay_buffer.sample()
                            agent.save_net()
                    else:
                        action = agent2.take_action(state)
                        next_state, reward, done = env.next(action)
                        n_step_counter -= 1
                        n_step_reward_cache.append(reward)
                        n_step_state_cache.append(state)
                        if n_step_counter == 0:
                            reward = 0
                            # nstep的動作狀態價值
                            for j in range(len(n_step_reward_cache)):
                                reward += n_step_reward_cache[j] * (agent2.gamma ** j)
                            replay_bufferN.add(n_step_state_cache[0], action, reward, next_state, done,
                                               n_steps[_type - 1])
                            cache.append((n_step_state_cache[0], action, reward, next_state, done,
                                          n_steps[_type - 1]))
                            n_step_counter = n_steps[_type - 1]
                            n_step_reward_cache.clear()
                            n_step_state_cache.clear()
                        elif done == 1:
                            reward = 0
                            # nstep的動作狀態價值
                            for j in range(len(n_step_reward_cache)):
                                reward += n_step_reward_cache[j] * (agent2.gamma ** j)
                            replay_bufferN.add(n_step_state_cache[0], action, reward, next_state, done,
                                               n_steps[_type - 1] - n_step_counter)
                            cache.append((n_step_state_cache[0], action, reward, next_state, done,
                                          n_steps[_type - 1] - n_step_counter))
                            n_step_counter = n_steps[_type - 1]
                            n_step_reward_cache.clear()
                            n_step_state_cache.clear()

                        total_step += 1
                        if replay_bufferN.size(
                        ) >= minimal_size and total_step % int(update_interval) == 0:
                            sample = replay_bufferN.sample()
                            agent2.save_net()
                    if done == 1:
                        break
                env.reset()
            with open("./cache/" + area + "_{}".format(_type), "w") as file:
                file.write(json.dumps(cache))
    # print(_result / _time)
