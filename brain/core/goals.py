"""
目标管理与规划系统 (Goal Management & Planning)

功能:
  - 目标栈: 当前活跃目标的层级结构
  - 目标分解: 大目标→子目标→可执行动作
  - 进度监控: 跟踪每个目标的完成度
  - 优先级调度: 根据紧急/重要矩阵排序
  - 中断处理: 新目标抢占、恢复被中断目标
  - 规划: 从当前状态到目标的路径搜索

与注意力联动: 目标驱动自上而下注意力
与动机联动: 目标进展产生内在奖励
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
import time as _time


class GoalStatus(Enum):
    PENDING = "pending"         # 待处理
    ACTIVE = "active"           # 进行中
    BLOCKED = "blocked"         # 阻塞
    COMPLETED = "completed"     # 已完成
    ABANDONED = "abandoned"     # 已放弃
    INTERRUPTED = "interrupted" # 被中断


class GoalPriority(Enum):
    LOW = (1, "低")
    MEDIUM = (2, "中")
    HIGH = (3, "高")
    CRITICAL = (4, "紧急")

    def __init__(self, level, label):
        self.level = level
        self.label = label


@dataclass
class Subgoal:
    """子目标"""
    id: str
    description: str
    completed: bool = False
    progress: float = 0.0
    order: int = 0


@dataclass
class Goal:
    """目标"""
    id: str
    description: str
    priority: GoalPriority = GoalPriority.MEDIUM
    status: GoalStatus = GoalStatus.PENDING
    parent_id: Optional[str] = None
    subgoals: List[Subgoal] = field(default_factory=list)
    progress: float = 0.0
    created_at: float = 0.0
    deadline: Optional[float] = None
    urgency: float = 0.5       # 紧急度 0-1
    importance: float = 0.5    # 重要度 0-1
    attempts: int = 0
    completion_time: Optional[float] = None
    reward_on_complete: float = 0.5
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = _time.time()

    @property
    def is_done(self) -> bool:
        return self.status == GoalStatus.COMPLETED

    def update_progress(self):
        """根据子目标更新进度"""
        if not self.subgoals:
            return
        completed = sum(1 for s in self.subgoals if s.completed)
        self.progress = completed / len(self.subgoals)
        if self.progress >= 1.0:
            self.status = GoalStatus.COMPLETED
            self.completion_time = _time.time()


class GoalManager:
    """
    目标管理器

    维护目标栈, 支持分解、优先级、中断恢复。
    """

    def __init__(self, max_stack: int = 10):
        self.goals: Dict[str, Goal] = {}
        self.goal_stack: List[str] = []  # 目标栈(栈顶=当前目标)
        self.max_stack = max_stack
        self.completed_goals: List[str] = []
        self.abandoned_goals: List[str] = []

        # 规划
        self.action_plan: List[str] = []  # 当前行动计划
        self.plan_index: int = 0

        # 统计
        self.total_completed = 0
        self.total_abandoned = 0
        self.total_interrupts = 0

    def add_goal(self, goal_id: str, description: str,
                 priority: GoalPriority = GoalPriority.MEDIUM,
                 parent_id: str = None,
                 urgency: float = 0.5, importance: float = 0.5,
                 subgoals: List[str] = None,
                 reward: float = 0.5,
                 tags: List[str] = None) -> Goal:
        """添加新目标"""
        goal = Goal(
            id=goal_id,
            description=description,
            priority=priority,
            parent_id=parent_id,
            urgency=urgency,
            importance=importance,
            reward_on_complete=reward,
            tags=tags or [],
        )

        if subgoals:
            for i, sg_desc in enumerate(subgoals):
                goal.subgoals.append(Subgoal(
                    id=f"{goal_id}_sg{i}",
                    description=sg_desc,
                    order=i,
                ))

        self.goals[goal_id] = goal

        # 加入目标栈(按优先级插入)
        if parent_id is None:
            self._push_goal(goal_id)

        return goal

    def _push_goal(self, goal_id: str):
        """将目标压入栈(按优先级排序)"""
        if goal_id in self.goal_stack:
            return
        if len(self.goal_stack) >= self.max_stack:
            # 栈满, 放弃最低优先级
            self._abandon_lowest_priority()
        self.goal_stack.append(goal_id)
        self.goals[goal_id].status = GoalStatus.ACTIVE
        self._sort_stack()

    def _sort_stack(self):
        """按优先级和紧急度排序目标栈"""
        def priority_score(gid):
            g = self.goals[gid]
            return (g.priority.level * 2 + g.urgency + g.importance * 0.5)
        self.goal_stack.sort(key=priority_score, reverse=True)

    def _abandon_lowest_priority(self):
        """放弃最低优先级目标"""
        if not self.goal_stack:
            return
        lowest = self.goal_stack[-1]
        self.goals[lowest].status = GoalStatus.ABANDONED
        self.abandoned_goals.append(lowest)
        self.goal_stack.pop()
        self.total_abandoned += 1

    def get_current_goal(self) -> Optional[Goal]:
        """获取当前目标(栈顶)"""
        if not self.goal_stack:
            return None
        return self.goals[self.goal_stack[0]]

    def interrupt_with(self, goal_id: str) -> Optional[str]:
        """
        用新目标中断当前目标

        Returns:
            被中断的目标ID(如果有)
        """
        interrupted = None
        if self.goal_stack:
            current = self.goal_stack[0]
            if self.goals[current].priority.level < self.goals[goal_id].priority.level:
                self.goals[current].status = GoalStatus.INTERRUPTED
                interrupted = current
                self.total_interrupts += 1

        self._push_goal(goal_id)
        return interrupted

    def complete_subgoal(self, goal_id: str, subgoal_id: str):
        """完成子目标"""
        if goal_id not in self.goals:
            return
        goal = self.goals[goal_id]
        for sg in goal.subgoals:
            if sg.id == subgoal_id:
                sg.completed = True
                sg.progress = 1.0
                break
        goal.update_progress()
        if goal.is_done:
            self._complete_goal(goal_id)

    def update_progress(self, goal_id: str, progress: float):
        """直接更新目标进度"""
        if goal_id in self.goals:
            goal = self.goals[goal_id]
            goal.progress = min(1.0, progress)
            if goal.progress >= 1.0:
                self._complete_goal(goal_id)

    def _complete_goal(self, goal_id: str):
        """完成目标"""
        goal = self.goals[goal_id]
        goal.status = GoalStatus.COMPLETED
        goal.completion_time = _time.time()
        if goal_id in self.goal_stack:
            self.goal_stack.remove(goal_id)
        self.completed_goals.append(goal_id)
        self.total_completed += 1

        # 恢复被中断的目标
        interrupted = [gid for gid in self.goal_stack
                      if self.goals[gid].status == GoalStatus.INTERRUPTED]
        if interrupted:
            self.goals[interrupted[0]].status = GoalStatus.ACTIVE

    def abandon_goal(self, goal_id: str):
        """放弃目标"""
        if goal_id in self.goals:
            self.goals[goal_id].status = GoalStatus.ABANDONED
            if goal_id in self.goal_stack:
                self.goal_stack.remove(goal_id)
            self.abandoned_goals.append(goal_id)
            self.total_abandoned += 1

    def make_plan(self, goal_id: str, actions: List[str]):
        """制定行动计划"""
        self.action_plan = actions
        self.plan_index = 0
        if goal_id in self.goals:
            self.goals[goal_id].attempts += 1

    def next_action(self) -> Optional[str]:
        """获取计划中的下一个动作"""
        if self.plan_index < len(self.action_plan):
            action = self.action_plan[self.plan_index]
            self.plan_index += 1
            return action
        return None

    def get_attention_bias(self) -> Tuple[Optional[str], float]:
        """
        获取自上而下注意力偏置(给注意力系统)

        Returns:
            (目标描述, 注意力强度)
        """
        current = self.get_current_goal()
        if current is None:
            return None, 0.0
        strength = current.importance * 0.5 + current.urgency * 0.5
        return current.description, float(np.clip(strength, 0, 1))

    def get_reward_signal(self) -> float:
        """
        获取目标完成奖励信号(给RL系统)

        Returns:
            近期完成目标的奖励
        """
        if not self.completed_goals:
            return 0.0
        recent = self.completed_goals[-3:]
        rewards = [self.goals[gid].reward_on_complete for gid in recent]
        return float(np.mean(rewards))

    def get_active_goals(self) -> List[Dict]:
        """获取所有活跃目标"""
        return [
            {
                "id": gid,
                "description": self.goals[gid].description,
                "priority": self.goals[gid].priority.label,
                "progress": round(self.goals[gid].progress, 2),
                "urgency": round(self.goals[gid].urgency, 2),
                "status": self.goals[gid].status.value,
            }
            for gid in self.goal_stack
        ]

    def step(self) -> Dict:
        """
        目标管理步进

        - 更新紧急度(随时间)
        - 检查阻塞
        - 返回当前状态
        """
        current = self.get_current_goal()
        for gid in self.goal_stack:
            goal = self.goals[gid]
            # 紧急度随截止时间临近上升
            if goal.deadline:
                remaining = max(0, goal.deadline - _time.time())
                goal.urgency = float(np.clip(1.0 - remaining / 86400, 0, 1))

        return {
            "current_goal": current.description if current else None,
            "stack_depth": len(self.goal_stack),
            "active_goals": len(self.goal_stack),
            "plan_progress": (self.plan_index / len(self.action_plan)
                            if self.action_plan else 0),
            "total_completed": self.total_completed,
        }

    def get_summary(self) -> Dict:
        return {
            "active_goals": len(self.goal_stack),
            "completed": self.total_completed,
            "abandoned": self.total_abandoned,
            "interrupts": self.total_interrupts,
            "current": (self.get_current_goal().description
                       if self.get_current_goal() else None),
            "plan_remaining": len(self.action_plan) - self.plan_index,
        }
