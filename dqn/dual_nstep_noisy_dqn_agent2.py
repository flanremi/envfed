import math
import os

import numpy as np
import torch
import torch.nn.functional as F


current_path = os.path.dirname(__file__) + "/"

class NoisyLinear(torch.nn.Module):
    """Noisy linear module for NoisyNet.

    Attributes:
        in_features (int): input size of linear module
        out_features (int): output size of linear module
        std_init (float): initial std value
        weight_mu (nn.Parameter): mean value weight parameter
        weight_sigma (nn.Parameter): std value weight parameter
        bias_mu (nn.Parameter): mean value bias parameter
        bias_sigma (nn.Parameter): std value bias parameter

    """

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        """Initialization."""
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = torch.nn.Parameter(
            torch.Tensor(out_features, in_features)
        )
        self.register_buffer(
            "weight_epsilon", torch.Tensor(out_features, in_features)
        )

        self.bias_mu = torch.nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = torch.nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Reset trainable network parameters (factorized gaussian noise)."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features)
        )

    def reset_noise(self):
        """Make new noise."""
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation.

        We don't use separate statements on train / eval mode.
        It doesn't show remarkable difference of performance.
        """
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )

    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        """Set scale to make noise (factorized gaussian noise)."""
        x = torch.randn(size)

        return x.sign().mul(x.abs().sqrt())

class Qnet(torch.nn.Module):

    def __init__(self, device, input_dim, action_dim,):
        super().__init__()
        self.device = device
        hidden = 2 ** (int(math.log2(input_dim)))
        self.fc1 = torch.nn.Linear(input_dim, hidden).to(device)
        self.ln1 = torch.nn.LayerNorm(hidden).to(device)

        self.fc2 = torch.nn.Linear(hidden, int(hidden / 4)).to(device)
        self.ln2 = torch.nn.LayerNorm(int(hidden / 4)).to(device)

        self.fc3 = torch.nn.Linear(int(hidden / 4), int(hidden / 4)).to(device)
        self.ln3 = torch.nn.LayerNorm(int(hidden / 4)).to(device)

        self.advantage_hidden_layer = NoisyLinear(int(hidden / 4), int(hidden / 4)).to(device)
        self.advantage_layer = NoisyLinear(int(hidden / 4), action_dim).to(device)

        self.value_hidden_layer = NoisyLinear(int(hidden / 4), int(hidden / 4)).to(device)
        self.value_layer = NoisyLinear(int(hidden / 4), 1).to(device)
        # self.fc6 = torch.nn.Linear(hidden, action_space).to(device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.ln1(x)

        x = F.relu(self.fc2(x))
        x = self.ln2(x)

        x = F.relu(self.fc3(x))
        x = self.ln3(x)

        adv_hid = F.relu(self.advantage_hidden_layer(x))
        val_hid = F.relu(self.value_hidden_layer(x))

        value = self.value_layer(adv_hid)
        advantage = self.advantage_layer(val_hid)

        q = value + advantage - advantage.mean(dim=-1, keepdim=True)

        return q
        # return self.fc6(x)

    def reset_noise(self):
        """Reset all noisy layers."""
        self.advantage_hidden_layer.reset_noise()
        self.advantage_layer.reset_noise()
        self.value_hidden_layer.reset_noise()
        self.value_layer.reset_noise()

class DDQN:
    ''' DQN算法 '''

    def __init__(self, device, input_dim, action_dim, sigma = 0.01, actor_lr = 3e-4, learning_rate = 0.001,
                 gamma = 0.95, tau = 0.005, epsilon = 0.05, target_update = 10):
        self.action_dim = action_dim
        self.input_dim = input_dim
        self.q_net = Qnet(device, self.input_dim, self.action_dim) # Q网络
        # 目标网络
        self.target_q_net = Qnet(device, self.input_dim, self.action_dim)
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate)
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device

    def save_net(self):
        global current_path
        torch.save(self.q_net.state_dict(), current_path + "net/" + "dual_nstep_noisy_dqn_q_net2")
        torch.save(self.target_q_net.state_dict(), current_path + "net/" + "dual_nstep_noisy_dqn_target_q_net2")

    def load_net(self):
        global current_path
        self.q_net.load_state_dict(torch.load(current_path + "net/" + "dual_nstep_noisy_dqn_q_net2"))
        self.target_q_net.load_state_dict(torch.load(current_path + "net/" + "dual_nstep_noisy_dqn_target_q_net2"))

    def take_action(self, state, explore = True):  # epsilon-贪婪策略采取动作
        if explore:
            if np.random.random() < self.epsilon:
                action = np.random.randint(self.action_dim)
            else:
                state = torch.tensor([state], dtype=torch.float).to(self.device)
                action = self.q_net(state).argmax().item()
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()

        return action

    def take_action2(self, state):  # 无贪婪
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action = self.q_net(state).argmax().item()

        return action

    # Q是向后传递的，期望是向前迭代的
    def update(self, n_sample):

        def get_loss(transition_dict):
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
            steps = torch.tensor(transition_dict[5],
                                  dtype=torch.float).view(-1, 1).to(self.device)

            q_values = self.q_net(states).gather(1, actions)  # Q值
            # 下个状态的最大Q值
            # max_next_q_values = self.target_q_net(next_states).max(1)[0].view(
            #     -1, 1)
            # ddqn的不同
            actiont = self.q_net(next_states)
            action2 = actiont.argmax(dim=1).unsqueeze(dim=1)
            max_next_q_values = self.target_q_net(next_states).gather(1, action2)
            q_targets = rewards + (self.gamma ** steps) * max_next_q_values * (1 - dones
                                                                    )  # TD误差目标
            _dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
            return _dqn_loss
        # print("dqn_loss: " + str(dqn_loss.detach().cpu().item()))

        # print(dqn_loss)
        dqn_loss = get_loss(n_sample)
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        self.q_net.reset_noise()
        self.target_q_net.reset_noise()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # 更新目标网络
        self.count += 1

        return dqn_loss.detach().cpu().item()
