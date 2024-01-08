import json
import time

import numpy as np
import torch

from dqn.state_container import CacheContainer

import dqn.dqn_agent as net


from dqn.replay_buffer import ReplayBuffer, ReplayBufferN
import environment

num_episodes = 1000
buffer_size = 10000

minimal_size = 128
batch_size = 64
update_interval = 10

if __name__ == '__main__':
    # sigma = 0.3  # 高斯噪声标准差
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    total_step = 0
    env = environment.Environment(environment.Type.crossing.value)
    replay_buffer = ReplayBuffer(buffer_size, batch_size)

    agent = net.DDQN(device, 2 + 25 * 25, 25 + 1)

    return_list = []

    loss_data = []
    reward_data = []
    steps = []

    for i_episode in range(num_episodes):
        env.reset()
        # 转换下agent的位置，避免序号对训练的影响
        # ep_returns = np.zeros(len(env.agents))
        _step = 0
        while True:
            state = env.get_state()
            decision_time = time.time()
            action = agent.take_action(state)
            next_state, reward, done, valid = env.next(action, time.time() - decision_time)
            if valid:
                _step += 1
            replay_buffer.add(state,action,next_state,reward,done)


            if replay_buffer.size(
            ) >= minimal_size and total_step % int(update_interval) == 0:
                sample = replay_buffer.sample()
                # def stack_array(x):
                #     rearranged = [[sub_x[i] for sub_x in x]
                #                   for i in range(len(x[0]))]
                #     return [
                #         torch.FloatTensor(np.vstack(aa)).to(device)
                #         for aa in rearranged
                #     ]
                #
                #
                # sample = [stack_array(x) for x in sample]
                for a_i in range(1):
                    agent.update(sample)
                    agent.save_net()
                # print("train_time:" + str(time.time() - _t))
            if done == 1:
                reward_data.append(env.last_reward)
                loss_data.append(env.now_loss)
                steps.append(_step)
                print(str(i_episode) + "=============" + str(env.last_reward))
                break

    # print(_result / _time)
