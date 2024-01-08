import json
import time

import numpy as np
import torch

from state_container import CacheContainer

import dual_nstep_noisy_dqn_agent2 as net


from replay_buffer import ReplayBuffer, ReplayBufferN

num_episodes = 5000
buffer_size = 300000

minimal_size = 4096
batch_size = 1024
episode_length = 9  # 每条序列的最大长度
update_interval = 16


# sigma = 0.3  # 高斯噪声标准差
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
total_step = 0
env = SatEnv()
replay_buffer = ReplayBuffer(buffer_size, batch_size)
n_replay_buffer = ReplayBuffer(buffer_size, batch_size)
agent = net.DDQN(device)
cacheHelper = CacheContainer(replay_buffer, n_replay_buffer, 1)
env2 = SatEnv()
env2.taskCreator.loadData()
return_list = []

def evaluation():
    global env2
    total_reward_ = 0
    env2.refresh2()
    env2.next_time2()
    state_, tasks_ = env2.getStates(1)
    for e_i_ in range(episode_length):
        while len(state_) > 0:
            action_ = agent.take_action2(state_)
            actions_ = [[1 if action_ == i else 0
            for i in range(net.action_space)]]
            # sats_poses, sat_types = states2input(states)
            next_state_, reward_, done_, subtasks_, nofity_next_states_ = env2.step(actions_, state_, tasks_)
            for t_r_ in reward_:
                total_reward_ += t_r_
            state_, tasks_ = env2.getStates(1)

        env2.next_time2()
        state_, tasks_ = env2.getStates(1)
    print(total_reward_ / env2.taskCreator.task_num)
    if total_reward_ / env2.taskCreator.task_num > 7:
        print(1)

#  todo 校验全过程是否存在异常 训练结果还不错
train_name = "train_data/train_dual_nstep_noisy2_dqn_"

loss_data = []
reward_data = []
for i_episode in range(num_episodes):
    env.refresh()
    cacheHelper.refresh()
    env.next_time()
    # 转换下agent的位置，避免序号对训练的影响
    state, tasks = env.getStates(1)
    # ep_returns = np.zeros(len(env.agents))
    total_step += 1
    total_reward = 0
    _result = 0
    _start = 10
    _time = 0
    for e_i in range(episode_length):
        while len(state) > 0:
            _t = time.time()
            action = agent.take_action(state)
            if _start > 0 :
                _start -= 1
            else:
                _result += time.time() - _t
                _time += 1
            # sats_poses, sat_types = states2input(states)
            actions = [[1 if action == i else 0
            for i in range(net.action_space)]]
            next_state, reward, done, subtasks, nofity_next_states = env.step(actions, state, tasks)

            for t_r in reward:
                total_reward += t_r

            # replay_buffer.add(state, actions, reward, next_state, done)
            t = cacheHelper.packCache(state[0], action, reward[0], next_state[0], done[0], subtasks[0])
            cacheHelper.appendCache(t[0], t[1])
            # 把queue中下一个组的数据读出来
            state, tasks = env.getStates(1)

        env.next_time()
        state, tasks = env.getStates(1)

        if replay_buffer.size(
        ) >= minimal_size and total_step % int(update_interval) == 0:
            sample = replay_buffer.sample()
            n_sample = n_replay_buffer.sample()

            _t = time.time()

            def stack_array(x):
                rearranged = [[sub_x[i] for sub_x in x]
                              for i in range(len(x[0]))]
                return [
                    torch.FloatTensor(np.vstack(aa)).to(device)
                    for aa in rearranged
                ]


            sample = [stack_array(x) for x in sample]
            n_sample = [stack_array(x) for x in n_sample]
            for a_i in range(1):
                loss_data.append(agent.update(sample, n_sample))
                # agent.save_net()
            # print("train_time:" + str(time.time() - _t))

    if total_step % update_interval ==0:
        print("=============")
        for i in range(1):
            evaluation()
    reward_data.append(total_reward / env.taskCreator.task_num)
    # print(_result / _time)

# with open(train_name + "batch_1024", "w") as file:
#     file.write(json.dumps((loss_data, reward_data)))
