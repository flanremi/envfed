import json
import time

import numpy as np
import torch

from dqn.state_container import CacheContainer

import dqn.dqn_agent as net
import dqn.dual_nstep_noisy_dqn_agent2 as net2


from dqn.replay_buffer import ReplayBuffer, ReplayBufferN
import environment

num_episodes = 500
buffer_size = 100000

minimal_size = 256
batch_size = 128
update_interval = 32
# 1， 3， 5
n_steps = [1, 3, 5]


if __name__ == '__main__':
    for _type in [1,2,3,0]:
        lam = [0.1, 0.3, 0.5, 0.7, 0.9]
        for la in lam:
            area = environment.Type.crossing.value
            # sigma = 0.3  # 高斯噪声标准差
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            total_step = 0
            env = environment.Environment(area)
            env.lamda = la

            replay_buffer = ReplayBuffer(buffer_size, batch_size)
            replay_bufferN = ReplayBufferN(buffer_size, batch_size)
            agent = net.DDQN(device, 3 + 16 * (25 + 1), 25 + 1, area + "_6_" + str(la), epsilon=0.10)
            agent2 = net2.DDQN(device, 3 + 16 * (25 + 1), 25 + 1, area + "_6_" + str(la), epsilon=0.10)

            return_list = []
            loss_data = []
            latency_data =[]
            reward_data = []
            steps = []
            dqn_loss = []

            n_step_reward_cache = []
            n_step_state_cache = []
            n_step_counter = n_steps[_type - 1]

            for i_episode in range(num_episodes):
                env.reset()
                # 转换下agent的位置，避免序号对训练的影响
                # ep_returns = np.zeros(len(env.agents))
                _step = 0
                while True:
                    state = env.get_state()
                    decision_time = time.time()
                    if _type == 0:
                        action = agent.take_action(state)
                        next_state, reward, done, valid = env.next(action, time.time() - decision_time)
                        if valid:
                            _step += 1
                        replay_buffer.add(state,action,reward,next_state,done)
                        total_step += 1
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
                                dqn_loss.append(agent.update(sample))
                                agent.save_net()
                            # print("train_time:" + str(time.time() - _t))
                        if done == 1:
                            reward_data.append(env.last_reward)
                            loss_data.append(env.now_loss)
                            steps.append(_step)
                            latency_data.append(env.latency)
                            print(str(_type) + "=============" + str(i_episode) + "=============" + str(env.last_reward))
                            with open("C:\\Users\\lily\\PycharmProjects\\zhangruoyi\\yolov5\\dqn\\{}_{}_result_6".format(area, la), "w+") as file:
                                file.write(json.dumps({"reward":reward_data, "loss":loss_data, "steps":steps,
                                                       "latency":latency_data,"dqn_loss":dqn_loss}))
                            break
                    else:
                        action = agent2.take_action(state)
                        next_state, reward, done, valid = env.next(action, time.time() - decision_time)
                        if valid:
                            _step += 1
                        n_step_counter -= 1
                        n_step_reward_cache.append(reward)
                        n_step_state_cache.append(state)
                        if n_step_counter == 0:
                            reward = 0
                            # nstep的動作狀態價值
                            for j in range(len(n_step_reward_cache)):
                                reward += n_step_reward_cache[j] * ( agent2.gamma ** j )
                            replay_bufferN.add(n_step_state_cache[0],action,reward,next_state,done, n_steps[_type - 1])
                            n_step_counter = n_steps[_type - 1]
                            n_step_reward_cache.clear()
                            n_step_state_cache.clear()
                        elif done == 1:
                            reward = 0
                            # nstep的動作狀態價值
                            for j in range(len(n_step_reward_cache)):
                                reward += n_step_reward_cache[j] * (agent2.gamma ** j)
                            replay_bufferN.add(n_step_state_cache[0],action,reward,next_state,done, n_steps[_type - 1] - n_step_counter)
                            n_step_counter = n_steps[_type - 1]
                            n_step_reward_cache.clear()
                            n_step_state_cache.clear()


                        total_step += 1
                        if replay_bufferN.size(
                        ) >= minimal_size and total_step % int(update_interval) == 0:
                            sample = replay_bufferN.sample()
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
                                dqn_loss.append(agent2.update(sample))
                                agent2.save_net()
                            # print("train_time:" + str(time.time() - _t))
                        if done == 1:
                            reward_data.append(env.last_reward)
                            loss_data.append(env.now_loss)
                            steps.append(_step)
                            latency_data.append(env.latency)
                            print(str(_type) + "=============" + str(i_episode) + "=============" + str(env.last_reward))
                            with open("C:\\Users\\lily\\PycharmProjects\\zhangruoyi\\yolov5\\dqn\\{}_{}_{}step_result_6"
                                              .format(area, la, n_steps[_type - 1]), "w+") as file:
                                file.write(json.dumps({"reward":reward_data, "loss":loss_data, "steps":steps,
                                                       "latency":latency_data, "dqn_loss":dqn_loss}))
                            break

    # print(_result / _time)
