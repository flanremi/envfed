import numpy as np
from enum import Enum
class Catalog(Enum):
    state = "state"
    action = "action"
    reward = "reward"
    next_state = "next_state"
    done = "done"
    step = "step"
    eid = "eid"

# 支持多步计算
class CacheContainer:
    # 使用单双2种数据集结合训练
    def __init__(self, replay_buffer, n_replay_buffer, gamma, step ) -> None:
        super().__init__()
        # 存储每一个cache信息，subtask_id为主键
        self.data = {}
        self.gamma = gamma
        self.step = step

        self.replay_buffer = replay_buffer
        self.n_replay_buffer = n_replay_buffer

    def refresh(self):
        self.data.clear()

    # 蒙特卡洛法是不需要记录每一次操作的，因为蒙特卡洛法就是通过多次采样，从而得到某一状态的估计值。
    # step从0开始
    def appendCache(self, cache:dict):
        if self.data.get(cache[Catalog.eid.value]) is None:
            self.data.update({cache[Catalog.eid.value]: cache})
        else:
            first_cache = self.data.get(cache[Catalog.eid.value])
            reward = first_cache.get(Catalog.reward.value) + (self.gamma ** (cache[Catalog.step.value] - first_cache[Catalog.step.value]))* cache.get(Catalog.reward.value)
            first_cache.update({Catalog.reward.value: reward})
        self.replay_buffer.add(
            cache[Catalog.state.value],
            cache[Catalog.action.value],
            cache[Catalog.reward.value],
            cache[Catalog.next_state.value],
            cache[Catalog.done.value],
        )
        first_cache = self.data.get(cache[Catalog.eid.value])
        # 序号从0开始，因此step-1, 保存N步数据
        if cache[Catalog.step.value] - first_cache[Catalog.step.value] >= self.step - 1 or cache[Catalog.done.value] == 0:
            first_cache = self.data.pop(cache[Catalog.eid.value])
            self.n_replay_buffer.add(
                first_cache[Catalog.state.value],
                first_cache[Catalog.action.value],
                first_cache[Catalog.reward.value],
                first_cache[Catalog.next_state.value],
                first_cache[Catalog.done.value],
                cache[Catalog.step.value] - first_cache[Catalog.step.value]
            )

