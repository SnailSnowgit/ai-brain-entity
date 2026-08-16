"""
强化学习完整回路 (Reinforcement Learning)

Actor-Critic架构, 多巴胺RPE驱动:
  - Critic: 状态价值估计 V(s)
  - Actor: 策略网络 π(a|s)
  - 多巴胺信号: TD误差 = R + γV(s') - V(s)
  - 资格迹(Eligibility Traces): 信用分配
  - 内在奖励: 好奇心+预测误差+稳态需求

与基底神经节联动:
  - Actor对应纹状体D1(Go)/D2(NoGo)通路
  - Critic对应多巴胺价值估计
  - 多巴胺RPE驱动两者更新
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto


class RewardSource(Enum):
    """奖励来源"""
    EXTRINSIC = "extrinsic"       # 外部奖励
    CURIOSITY = "curiosity"       # 好奇心(预测误差)
    HOMEOSTATIC = "homeostatic"   # 稳态(需求满足)
    SOCIAL = "social"             # 社交
    ACHIEVEMENT = "achievement"   # 成就/目标
    NOVELTY = "novelty"           # 新颖性


@dataclass
class Transition:
    """状态转移"""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    info: Dict = field(default_factory=dict)


class EligibilityTrace:
    """资格迹(指数衰减)"""

    def __init__(self, dims: Tuple[int, ...], decay: float = 0.9):
        self.trace = np.zeros(dims)
        self.decay = decay

    def update(self, gradient: np.ndarray):
        """累积梯度"""
        self.trace = self.decay * self.trace + gradient

    def decay_step(self):
        """衰减"""
        self.trace *= self.decay

    def reset(self):
        self.trace.fill(0)


class CriticNetwork:
    """
    Critic: 状态价值网络 V(s)

    线性近似 + 资格迹
    """

    def __init__(self, state_dim: int = 128, lr: float = 0.01,
                 gamma: float = 0.95, trace_decay: float = 0.9):
        self.state_dim = state_dim
        self.lr = lr
        self.gamma = gamma

        # V(s) = w · φ(s)
        self.weights = np.random.randn(state_dim) * 0.05
        self.eligibility = EligibilityTrace((state_dim,), trace_decay)

        self.value_history: List[float] = []

    def value(self, state: np.ndarray) -> float:
        """估计状态价值"""
        return float(np.tanh(state @ self.weights))

    def td_error(self, reward: float, state: np.ndarray,
                 next_state: np.ndarray, done: bool = False) -> float:
        """计算TD误差(多巴胺RPE)"""
        v_current = self.value(state)
        v_next = 0.0 if done else self.value(next_state)
        td = reward + self.gamma * v_next - v_current
        return float(td)

    def learn(self, state: np.ndarray, td_error: float):
        """更新价值网络(带资格迹)"""
        # 特征 = 状态本身(线性近似)
        grad = state
        self.eligibility.update(grad)
        self.weights += self.lr * td_error * self.eligibility.trace
        self.weights = np.clip(self.weights, -5, 5)
        self.eligibility.decay_step()

        self.value_history.append(self.value(state))
        if len(self.value_history) > 500:
            self.value_history.pop(0)


class ActorNetwork:
    """
    Actor: 策略网络 π(a|s)

    softmax策略 + 资格迹
    """

    def __init__(self, state_dim: int = 128, n_actions: int = 10,
                 lr: float = 0.02, trace_decay: float = 0.85,
                 entropy_beta: float = 0.05):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.lr = lr
        self.entropy_beta = entropy_beta

        # 策略权重 θ: 状态→动作偏好
        self.weights = np.random.randn(state_dim, n_actions) * 0.05
        self.eligibility = EligibilityTrace((state_dim, n_actions), trace_decay)

        self.action_history: List[int] = []

    def _preferences(self, state: np.ndarray) -> np.ndarray:
        """动作偏好"""
        return state @ self.weights

    def policy(self, state: np.ndarray) -> np.ndarray:
        """softmax策略"""
        prefs = self._preferences(state)
        prefs = prefs - prefs.max()  # 数值稳定
        exp_prefs = np.exp(prefs)
        return exp_prefs / exp_prefs.sum()

    def select_action(self, state: np.ndarray,
                      temperature: float = 1.0) -> Tuple[int, float]:
        """
        选择动作

        Returns:
            (action_idx, probability)
        """
        probs = self.policy(state)
        # 温度调节
        log_probs = np.log(probs + 1e-8) / temperature
        exp_probs = np.exp(log_probs - log_probs.max())
        probs = exp_probs / exp_probs.sum()

        action = np.random.choice(self.n_actions, p=probs)
        self.action_history.append(action)
        if len(self.action_history) > 500:
            self.action_history.pop(0)
        return int(action), float(probs[action])

    def learn(self, state: np.ndarray, action: int, td_error: float):
        """策略梯度更新(带资格迹)"""
        probs = self.policy(state)

        # 策略梯度: ∇log π(a|s)
        grad = np.zeros((self.state_dim, self.n_actions))
        for a in range(self.n_actions):
            indicator = 1.0 if a == action else 0.0
            grad[:, a] = state * (indicator - probs[a])

        # 熵正则化(鼓励探索)
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        entropy_grad = np.zeros((self.state_dim, self.n_actions))
        for a in range(self.n_actions):
            entropy_grad[:, a] = -state * (np.log(probs[a] + 1e-8) + 1)

        self.eligibility.update(grad)
        self.weights += (self.lr * td_error * self.eligibility.trace +
                        self.entropy_beta * entropy_grad)
        self.weights = np.clip(self.weights, -5, 5)
        self.eligibility.decay_step()

    def greedy_action(self, state: np.ndarray) -> int:
        """贪婪动作(利用)"""
        return int(np.argmax(self._preferences(state)))


class ReinforcementLearning:
    """
    完整强化学习回路

    整合Actor-Critic、多巴胺RPE、内在奖励、资格迹。
    """

    def __init__(self,
                 state_dim: int = 128,
                 n_actions: int = 10,
                 actor_lr: float = 0.02,
                 critic_lr: float = 0.01,
                 gamma: float = 0.95,
                 trace_decay: float = 0.9,
                 curiosity_weight: float = 0.3,
                 novelty_weight: float = 0.2,
                 homeostatic_weight: float = 0.5):
        self.critic = CriticNetwork(state_dim, critic_lr, gamma, trace_decay)
        self.actor = ActorNetwork(state_dim, n_actions, actor_lr, trace_decay)
        self.gamma = gamma

        # 内在奖励权重
        self.curiosity_weight = curiosity_weight
        self.novelty_weight = novelty_weight
        self.homeostatic_weight = homeostatic_weight

        # 经验回放
        self.replay_buffer: List[Transition] = []
        self.max_buffer = 1000

        # 新颖性记录(访问过的状态)
        self.visited_states: List[np.ndarray] = []
        self.max_visited = 500

        # 统计
        self.total_steps = 0
        self.total_reward = 0.0
        self.episode_rewards: List[float] = []
        self.current_episode_reward = 0.0
        self.dopamine_history: List[float] = []
        self.reward_breakdown: Dict[str, float] = {
            s.value: 0.0 for s in RewardSource
        }

    def compute_intrinsic_reward(self, state: np.ndarray,
                                 prediction_error: float = 0.0,
                                 homeostatic_signals: Dict = None,
                                 social: bool = False) -> Dict[str, float]:
        """
        计算内在奖励

        Returns:
            各来源奖励dict
        """
        rewards = {}

        # 好奇心奖励(预测误差, 金发姑娘区: 不要太大也不要太小)
        curiosity = 0.0
        if 0.05 < prediction_error < 0.8:
            curiosity = prediction_error * (1 - prediction_error) * 4
        rewards[RewardSource.CURIOSITY.value] = curiosity * self.curiosity_weight

        # 新颖性奖励(未访问过的状态)
        novelty = 0.0
        if self.visited_states:
            similarities = [float(np.abs(np.dot(state, v)))
                           for v in self.visited_states[-50:]]
            novelty = 1.0 - max(similarities)
        else:
            novelty = 1.0
        rewards[RewardSource.NOVELTY.value] = novelty * self.novelty_weight

        # 稳态奖励(需求满足)
        homeostatic = 0.0
        if homeostatic_signals:
            # 稳态信号是缺失度, 奖励 = 缺失减少
            for need, deficit in homeostatic_signals.items():
                homeostatic += max(0, 0.1 - deficit)
            homeostatic /= max(1, len(homeostatic_signals))
        rewards[RewardSource.HOMEOSTATIC.value] = homeostatic * self.homeostatic_weight

        # 社交奖励
        rewards[RewardSource.SOCIAL.value] = 0.2 if social else 0.0

        return rewards

    def step(self, state: np.ndarray,
             extrinsic_reward: float = 0.0,
             prediction_error: float = 0.0,
             homeostatic_signals: Dict = None,
             social: bool = False,
             temperature: float = 1.0,
             done: bool = False) -> Dict:
        """
        RL完整一步

        Args:
            state: 当前状态
            extrinsic_reward: 外部奖励
            prediction_error: 预测编码误差(用于好奇心)
            homeostatic_signals: 稳态信号
            social: 是否社交
            temperature: 探索温度
            done: 是否回合结束

        Returns:
            {action, action_prob, dopamine, total_reward, rewards}
        """
        # 选择动作
        action, action_prob = self.actor.select_action(state, temperature)

        # 计算内在奖励
        intrinsic = self.compute_intrinsic_reward(
            state, prediction_error, homeostatic_signals, social)

        # 总奖励
        total_reward = extrinsic_reward + sum(intrinsic.values())
        rewards = {RewardSource.EXTRINSIC.value: extrinsic_reward, **intrinsic}

        # 记录
        for k, v in rewards.items():
            self.reward_breakdown[k] += v
        self.total_reward += total_reward
        self.current_episode_reward += total_reward

        # 记录状态访问
        self.visited_states.append(state.copy())
        if len(self.visited_states) > self.max_visited:
            self.visited_states.pop(0)

        return {
            "action": action,
            "action_prob": action_prob,
            "total_reward": total_reward,
            "rewards": rewards,
        }

    def learn(self, state: np.ndarray, action: int, reward: float,
              next_state: np.ndarray, done: bool = False):
        """
        学习更新(Actor-Critic)

        在获得next_state后调用
        """
        # TD误差(多巴胺信号)
        td_error = self.critic.td_error(reward, state, next_state, done)
        self.dopamine_history.append(td_error)
        if len(self.dopamine_history) > 500:
            self.dopamine_history.pop(0)

        # 更新Critic
        self.critic.learn(state, td_error)

        # 更新Actor
        self.actor.learn(state, action, td_error)

        # 存储转移
        self.replay_buffer.append(Transition(
            state=state.copy(), action=action, reward=reward,
            next_state=next_state.copy(), done=done))
        if len(self.replay_buffer) > self.max_buffer:
            self.replay_buffer.pop(0)

        self.total_steps += 1

        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0.0

        return td_error

    def replay(self, n_samples: int = 10):
        """经验回放学习"""
        if len(self.replay_buffer) < n_samples:
            return

        indices = np.random.choice(len(self.replay_buffer), n_samples, replace=False)
        for idx in indices:
            t = self.replay_buffer[idx]
            self.learn(t.state, t.action, t.reward, t.next_state, t.done)

    def get_dopamine(self) -> float:
        """获取当前多巴胺信号"""
        if not self.dopamine_history:
            return 0.0
        # 近期平均
        return float(np.mean(self.dopamine_history[-10:]))

    def get_summary(self) -> Dict:
        return {
            "total_steps": self.total_steps,
            "total_reward": round(self.total_reward, 3),
            "avg_episode_reward": round(
                np.mean(self.episode_rewards[-20:])
                if self.episode_rewards else 0, 3),
            "dopamine": round(self.get_dopamine(), 4),
            "value_estimate": round(
                self.critic.value_history[-1]
                if self.critic.value_history else 0, 3),
            "reward_breakdown": {
                k: round(v, 3) for k, v in self.reward_breakdown.items()
            },
            "replay_buffer": len(self.replay_buffer),
            "explored_states": len(self.visited_states),
        }
