"""
Brain Simulator v4.0
类脑认知架构 — 18大模块

核心模块:
  1. language_model      — 基础LLM后端
  2. memory              — 类脑三级记忆
  3. emotion             — 情绪核心+多巴胺
  4. consciousness       — 全局工作空间意识
  5. motivation          — 内在动机
  6. bus                 — 模块通信总线
  7. predictive_coding   — 预测编码网络
  8. thought             — 思考系统

扩展模块:
  9. basal_ganglia       — 基底神经节(动作选择+习惯形成)
 10. attention           — 注意力系统
 11. sleep               — 睡眠与记忆重放
 12. homeostasis         — 稳态调节(能量/疲劳/昼夜节律)
 13. default_mode        — 默认模式网络(想象力/心智游移)
 14. metacognition       — 元认知与自我模型
 15. cerebellum          — 小脑(时序协调+自动化)
 16. reinforcement       — 强化学习完整回路
 17. goals               — 目标管理与规划
 18. emotion_regulation  — 情绪调节

工具:
  - evolution            — 遗传算法
  - model_io             — 模型导入导出
"""

from .core.language_model import LanguageModel, MockLanguageModel
from .core.memory import MemorySystem, MemoryItem, SensoryBuffer, ShortTermMemory, LongTermMemory
from .core.emotion import EmotionalCore, EmotionalState, DopamineSystem
from .core.consciousness import (
    ConsciousnessSystem, GlobalWorkspace, ConsciousContent,
    ConsciousnessLevel, ConsciousnessMetrics
)
from .core.motivation import MotivationSystem, Drive, DriveType, CuriosityEngine
from .core.bus import MessageBus, Message, MessageType, BusModule
from .core.predictive_coding import PredictiveCodingNetwork, PCLayer
from .core.thought import (
    ThoughtSystem, ThoughtSpace, ThoughtStream, Thought, ThoughtType
)
from .core.basal_ganglia import BasalGanglia, Action, ActionType
from .core.attention import AttentionSystem, AttentionMode
from .core.sleep import SleepSystem, SleepStage
from .core.homeostasis import HomeostaticSystem, NeedType, CircadianRhythm
from .core.default_mode import DefaultModeNetwork, DMNTheme
from .core.metacognition import Metacognition, SelfModel, MetaMemory, ConfidenceLevel
from .core.cerebellum import Cerebellum, ActionSequence
from .core.reinforcement import ReinforcementLearning, RewardSource
from .core.goals import GoalManager, Goal, GoalStatus, GoalPriority
from .core.emotion_regulation import EmotionRegulation, RegulationStrategy
from .core.evolution import (
    GeneticAlgorithm, Genome, HyperParams,
    flatten_pc_weights, unflatten_pc_weights, apply_hyperparams,
)
from .core.model_io import export_model, load_model, restore_modules, count_parameters

__version__ = "4.0"

__all__ = [
    # LLM
    'LanguageModel', 'MockLanguageModel',
    # 记忆
    'MemorySystem', 'MemoryItem', 'SensoryBuffer', 'ShortTermMemory', 'LongTermMemory',
    # 情绪
    'EmotionalCore', 'EmotionalState', 'DopamineSystem',
    # 意识
    'ConsciousnessSystem', 'GlobalWorkspace', 'ConsciousContent',
    'ConsciousnessLevel', 'ConsciousnessMetrics',
    # 动机
    'MotivationSystem', 'Drive', 'DriveType', 'CuriosityEngine',
    # 总线
    'MessageBus', 'Message', 'MessageType', 'BusModule',
    # 预测编码
    'PredictiveCodingNetwork', 'PCLayer',
    # 思考
    'ThoughtSystem', 'ThoughtSpace', 'ThoughtStream', 'Thought', 'ThoughtType',
    # 基底神经节
    'BasalGanglia', 'Action', 'ActionType',
    # 注意力
    'AttentionSystem', 'AttentionMode',
    # 睡眠
    'SleepSystem', 'SleepStage',
    # 稳态
    'HomeostaticSystem', 'NeedType', 'CircadianRhythm',
    # 默认模式
    'DefaultModeNetwork', 'DMNTheme',
    # 元认知
    'Metacognition', 'SelfModel', 'MetaMemory', 'ConfidenceLevel',
    # 小脑
    'Cerebellum', 'ActionSequence',
    # 强化学习
    'ReinforcementLearning', 'RewardSource',
    # 目标
    'GoalManager', 'Goal', 'GoalStatus', 'GoalPriority',
    # 情绪调节
    'EmotionRegulation', 'RegulationStrategy',
    # 进化
    'GeneticAlgorithm', 'Genome', 'HyperParams',
    'flatten_pc_weights', 'unflatten_pc_weights', 'apply_hyperparams',
    # 模型导入导出
    'export_model', 'load_model', 'restore_modules', 'count_parameters',
]
