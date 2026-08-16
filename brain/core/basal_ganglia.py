"""
基底神经节 (Basal Ganglia)

功能:
  - 动作选择: Go/NoGo双通路竞争(直接/间接通路)
  - 习惯形成: 反复成功的行为从目标导向转为自动化习惯
  - 多巴胺门控: 多巴胺RPE驱动Q值更新和习惯强化
  - 与丘脑皮层联动: 选中动作放行, 其余抑制

双系统:
  - 目标导向系统(Goal-directed): 深思熟虑, 慢, 消耗认知资源
  - 习惯系统(Habitual): 自动触发, 快, 无意识
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto


class ActionType(Enum):
    """内置动作类型"""
    SPEAK = auto()            # 说话
    THINK_DEEP = auto()       # 深度思考(系统2)
    EXPLORE = auto()          # 探索/提问
    AVOID = auto()            # 回避/谨慎
    APPROACH = auto()         # 趋近
    RETRIEVE_MEMORY = auto()  # 记忆检索
    REST = auto()             # 休息
    SOCIALIZE = auto()        # 社交
    LEARN = auto()            # 学习
    IMAGINE = auto()          # 想象
    CUSTOM = auto()           # 自定义


@dataclass
class Action:
    """可执行动作"""
    id: str
    name: str
    action_type: ActionType
    q_value: float = 0.0          # Q值(目标导向系统)
    habit_strength: float = 0.0   # 习惯强度 0-1
    go_strength: float = 0.0      # 直接通路(Go)激活
    nogo_strength: float = 0.0    # 间接通路(NoGo)激活
    execution_count: int = 0      # 执行次数
    success_count: int = 0        # 成功次数
    last_executed: float = -100.0 # 上次执行时间
    context_vector: Optional[np.ndarray] = None  # 关联上下文

    @property
    def success_rate(self) -> float:
        if self.execution_count == 0:
            return 0.0
        return self.success_count / self.execution_count

    @property
    def is_habit(self) -> bool:
        """习惯强度超过阈值且成功率高"""
        return self.habit_strength > 0.7 and self.success_rate > 0.6


class BasalGanglia:
    """
    基底神经节

    直接通路(Go): 促进动作, D1受体, 多巴胺兴奋
    间接通路(NoGo): 抑制动作, D2受体, 多巴胺抑制
    黑质致密部(SNc): 多巴胺释放, RPE
    丘脑: 被选中动作放行到皮层
    """

    def __init__(self,
                 learning_rate: float = 0.1,
                 habit_threshold: float = 0.7,
                 habit_learning_rate: float = 0.05,
                 go_baseline: float = 0.3,
                 nogo_baseline: float = 0.3,
                 discount: float = 0.9,
                 epsilon: float = 0.1):
        self.lr = learning_rate
        self.habit_threshold = habit_threshold
        self.habit_lr = habit_learning_rate
        self.go_baseline = go_baseline
        self.nogo_baseline = nogo_baseline
        self.discount = discount
        self.epsilon = epsilon

        self.actions: Dict[str, Action] = {}
        self.current_time = 0.0
        self.last_action: Optional[str] = None
        self.last_state: Optional[np.ndarray] = None

        # 纹状体D1/D2权重(状态→Go/NoGo)
        self.state_dim = 128
        self.d1_weights = np.random.randn(self.state_dim, 32) * 0.05  # Go
        self.d2_weights = np.random.randn(self.state_dim, 32) * 0.05  # NoGo
        self.critic_weights = np.random.randn(self.state_dim) * 0.05  # 价值估计

        # 统计
        self.total_selections = 0
        self.habit_selections = 0
        self.deliberation_time = 0.0

    def register_action(self, action_id: str, name: str,
                        action_type: ActionType = ActionType.CUSTOM,
                        context: np.ndarray = None):
        """注册新动作"""
        if action_id not in self.actions:
            self.actions[action_id] = Action(
                id=action_id, name=name, action_type=action_type,
                context_vector=context,
            )

    def _compute_state_value(self, state: np.ndarray) -> float:
        """Critic: 估计状态价值"""
        return float(np.tanh(state @ self.critic_weights))

    def _compute_go_nogo(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """纹状体: 计算Go(D1)和NoGo(D2)激活"""
        go = np.tanh(state @ self.d1_weights)      # 直接通路
        nogo = np.tanh(state @ self.d2_weights)    # 间接通路
        return go, nogo

    def _action_salience(self, action: Action, state: np.ndarray,
                         dopamine: float, motivation: float) -> float:
        """
        计算动作显著性(丘脑皮层激活)

        salience = Go - NoGo + Q值 + 习惯强度 + 动机偏置
        多巴胺增强Go, 抑制NoGo
        """
        go = action.go_strength + self.go_baseline
        nogo = action.nogo_strength + self.nogo_baseline

        # 多巴胺调制: 高DA→Go增强, NoGo减弱
        da_effect = dopamine - 0.3
        go += 0.5 * da_effect
        nogo -= 0.3 * da_effect

        # Q值贡献(目标导向)
        q_contrib = 0.3 * action.q_value

        # 习惯贡献(习惯系统)
        habit_contrib = 0.4 * action.habit_strength

        # 动机偏置
        mot_contrib = 0.2 * motivation

        # 新颖性 bonus(久未执行的动作)
        novelty = 0.1 * np.exp(-0.01 * (self.current_time - action.last_executed))

        salience = go - nogo + q_contrib + habit_contrib + mot_contrib + novelty
        return float(salience)

    def select_action(self, state: np.ndarray,
                      dopamine: float = 0.3,
                      motivations: Dict[str, float] = None,
                      force_deliberate: bool = False) -> Tuple[str, float, bool]:
        """
        选择动作

        Args:
            state: 当前状态向量(128维)
            dopamine: 当前多巴胺水平
            motivations: 各动作类型的动机强度
            force_deliberate: 强制深思熟虑(不走习惯)

        Returns:
            (action_id, confidence, is_habit)
        """
        if not self.actions:
            return ("wait", 0.0, False)

        self.current_time += 1.0
        self.last_state = state.copy()

        # 计算Go/NoGo通路
        go_act, nogo_act = self._compute_go_nogo(state)

        # 计算每个动作的显著性
        saliences = {}
        for aid, action in self.actions.items():
            # 更新Go/NoGo强度(简化: 用状态点积上下文)
            if action.context_vector is not None:
                action.go_strength = float(np.tanh(
                    np.dot(state[:32], go_act) +
                    0.3 * np.dot(state, action.context_vector) / self.state_dim))
                action.nogo_strength = float(np.tanh(
                    np.dot(state[:32], nogo_act) * 0.5))

            mot = 0.0
            if motivations:
                type_key = action.action_type.name.lower()
                mot = motivations.get(type_key, 0.3)

            saliences[aid] = self._action_salience(
                action, state, dopamine, mot)

        # 习惯优先检查: 如果有强习惯且不强制深思
        used_habit = False
        if not force_deliberate:
            strong_habits = [
                (aid, a) for aid, a in self.actions.items()
                if a.is_habit and saliences[aid] > 0.3
            ]
            if strong_habits and np.random.random() < 0.8:
                # 习惯系统: 直接选最强习惯(快速, 无意识)
                best = max(strong_habits, key=lambda x: x[1].habit_strength)
                self.last_action = best[0]
                self.total_selections += 1
                self.habit_selections += 1
                return (best[0], 0.9, True)

        # 目标导向系统: softmax选择(深思熟虑)
        aids = list(saliences.keys())
        vals = np.array([saliences[a] for a in aids])

        # epsilon探索
        if np.random.random() < self.epsilon:
            chosen_idx = np.random.randint(len(aids))
        else:
            # softmax with temperature
            temp = max(0.1, 1.0 - dopamine * 0.5)  # 高多巴胺→低温度→利用
            exp_vals = np.exp((vals - vals.max()) / temp)
            probs = exp_vals / exp_vals.sum()
            chosen_idx = np.random.choice(len(aids), p=probs)

        chosen_id = aids[chosen_idx]
        confidence = float(np.max(vals) - np.mean(vals))
        self.last_action = chosen_id
        self.total_selections += 1

        return (chosen_id, confidence, False)

    def update(self, reward: float, next_state: np.ndarray = None):
        """
        更新: 多巴胺RPE驱动Q值和习惯更新

        Args:
            reward: 实际奖励
            next_state: 下一状态(用于Q-learning)
        """
        if self.last_action is None or self.last_state is None:
            return

        action = self.actions.get(self.last_action)
        if action is None:
            return

        # Q-learning TD更新
        current_v = self._compute_state_value(self.last_state)
        next_v = self._compute_state_value(next_state) if next_state is not None else 0.0
        td_target = reward + self.discount * next_v
        td_error = td_target - current_v

        action.q_value += self.lr * td_error
        action.q_value = float(np.clip(action.q_value, -1, 1))

        # Critic更新
        self.critic_weights += self.lr * td_error * self.last_state
        self.critic_weights = np.clip(self.critic_weights, -5, 5)

        # D1/D2权重更新(简化)
        if td_error > 0:
            # 正RPE: 增强Go
            self.d1_weights += self.lr * 0.1 * td_error * np.outer(
                self.last_state, np.ones(32))
        else:
            # 负RPE: 增强NoGo
            self.d2_weights += self.lr * 0.1 * abs(td_error) * np.outer(
                self.last_state, np.ones(32))

        # 习惯强化/消退
        action.execution_count += 1
        action.last_executed = self.current_time
        if reward > 0.3:
            action.success_count += 1
            action.habit_strength += self.habit_lr * (1 - action.habit_strength)
        elif reward < -0.2:
            action.habit_strength -= self.habit_lr * 0.5
        action.habit_strength = float(np.clip(action.habit_strength, 0, 1))

        # D1/D2权重裁剪
        self.d1_weights = np.clip(self.d1_weights, -3, 3)
        self.d2_weights = np.clip(self.d2_weights, -3, 3)

    def get_action_stats(self) -> Dict:
        """获取动作统计"""
        return {
            "total_actions": len(self.actions),
            "total_selections": self.total_selections,
            "habit_selections": self.habit_selections,
            "habit_ratio": (self.habit_selections / max(1, self.total_selections)),
            "actions": {
                aid: {
                    "name": a.name,
                    "q_value": round(a.q_value, 3),
                    "habit_strength": round(a.habit_strength, 3),
                    "success_rate": round(a.success_rate, 3),
                    "is_habit": a.is_habit,
                    "executions": a.execution_count,
                }
                for aid, a in self.actions.items()
            },
        }
