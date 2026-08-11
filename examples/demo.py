"""
类脑认知架构模拟器 - 基础演示脚本

演示内容：
1. 基本刺激-反应通路
2. 记忆形成与提取
3. 情绪与注意力调制
4. 多巴胺奖励与强化学习
"""

import numpy as np
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain import Brain


def demo_basic_stimulus_response():
    """演示1：基本刺激-反应通路"""
    print("=" * 60)
    print("演示1：基本刺激-反应通路")
    print("=" * 60)
    
    # 创建大脑
    brain = Brain(
        sensory_neurons=100,
        association_neurons=150,
        decision_neurons=8,
        action_names=["前进", "后退", "左转", "右转", "探索", "休息", "进食", "躲避"]
    )
    
    print(f"\n创建大脑成功：{brain}")
    
    # 生成一个随机刺激（模拟视觉输入）
    stimulus = np.zeros(100)
    stimulus[20:40] = 1.0  # 中间区域有刺激
    stimulus += np.random.normal(0, 0.2, 100)  # 加入噪声
    
    print(f"\n输入视觉刺激：强度={np.sum(stimulus):.1f}, 位置=20-40")
    
    # 输入刺激并运行几步
    brain.input_stimulus(stimulus, modality=0)
    
    for i in range(50):
        state = brain.step(dt=1.0)
        
        if i % 10 == 0:
            print(f"\n第 {i} 步 (t={state.time:.0f}ms):")
            print(f"  感官层活动率: {state.sensory_activity:.2f} Hz")
            print(f"  联想层活动率: {state.association_activity:.2f} Hz")
            print(f"  决策层活动率: {state.decision_activity:.2f} Hz")
            print(f"  当前决策: {state.decision[1]} (置信度: {state.decision[2]:.2f})")
            print(f"  激活概念数: {len(state.active_concepts)}")
    
    print(f"\n最终决策: {brain.get_current_state().decision[1]}")
    print(f"记忆状态: {brain.memory.get_memory_stats()}")
    
    return brain


def demo_memory_system():
    """演示2：记忆系统运作"""
    print("\n" + "=" * 60)
    print("演示2：三级记忆系统")
    print("=" * 60)
    
    brain = Brain()
    
    # 重复输入相同刺激，形成记忆
    print("\n--- 重复输入相同刺激，观察记忆形成 ---")
    
    pattern = np.zeros(100)
    pattern[30:50] = 1.5  # 特定模式
    
    for trial in range(10):
        brain.input_stimulus(pattern, modality=0)
        for _ in range(20):
            brain.step(dt=1.0)
        
        stats = brain.memory.get_memory_stats()
        print(f"  第 {trial+1} 次刺激后: "
              f"感官缓存={stats['sensory_buffer_count']}, "
              f"短期记忆={stats['stm_count']}, "
              f"长期记忆={stats['ltm_count']}")
    
    # 测试记忆提取
    print("\n--- 测试记忆提取 ---")
    
    # 用部分线索测试提取
    cue = np.zeros(100)
    cue[35:45] = 1.0  # 部分线索
    
    retrieved = brain.memory.retrieve(cue)
    if retrieved:
        print(f"  线索提取成功！")
        print(f"  记忆强度: {retrieved.strength:.3f}")
        print(f"  提取次数: {retrieved.retrieval_count}")
        print(f"  情绪效价: {retrieved.emotional_valence:.3f}")
    else:
        print("  未能提取到记忆")
    
    # 运行更长时间，观察记忆巩固
    print("\n--- 继续运行，观察记忆巩固 ---")
    for _ in range(200):
        brain.step(dt=1.0)
    
    stats = brain.memory.get_memory_stats()
    print(f"  运行200步后:")
    print(f"    感官缓存: {stats['sensory_buffer_count']} (应已清空)")
    print(f"    短期记忆: {stats['stm_count']}")
    print(f"    长期记忆: {stats['ltm_count']}")
    print(f"    长期记忆平均强度: {stats['ltm_avg_strength']:.3f}")
    print(f"    联想连接数: {stats['ltm_associations']}")
    
    return brain


def demo_emotion_and_attention():
    """演示3：情绪与注意力调制"""
    print("\n" + "=" * 60)
    print("演示3：情绪与注意力调制")
    print("=" * 60)
    
    brain = Brain()
    
    # 中性刺激
    print("\n--- 中性刺激下的状态 ---")
    neutral_stim = np.random.normal(0.5, 0.3, 100)
    brain.input_stimulus(neutral_stim, modality=0, reward=0.0)
    for _ in range(30):
        brain.step(dt=1.0)
    
    state = brain.get_current_state()
    print(f"  情绪状态: {state.emotional_state}")
    print(f"  多巴胺水平: {state.dopamine_level:.3f}")
    print(f"  注意力焦点: 位置 {state.attention_focus}")
    
    # 正性奖励刺激
    print("\n--- 给予正性奖励后的变化 ---")
    brain.reward(1.0)  # 大奖励
    for _ in range(20):
        brain.step(dt=1.0)
    
    state = brain.get_current_state()
    print(f"  情绪状态: {state.emotional_state}")
    print(f"  多巴胺水平: {state.dopamine_level:.3f} (上升)")
    print(f"  记忆调制增强: {state.memory_stats.get('ltm_avg_strength', 0):.3f}")
    
    # 负性惩罚刺激
    print("\n--- 给予负性惩罚后的变化 ---")
    brain.reward(-0.8)  # 惩罚
    for _ in range(20):
        brain.step(dt=1.0)
    
    state = brain.get_current_state()
    print(f"  情绪状态: {state.emotional_state}")
    print(f"  多巴胺水平: {state.dopamine_level:.3f} (下降)")
    
    # 注意力目标设置
    print("\n--- 设置注意力目标 ---")
    goal_pattern = np.zeros(100)
    goal_pattern[60:80] = 1.0  # 关注右侧区域
    brain.set_attention_goal(goal_pattern)
    
    # 输入包含目标的刺激
    test_stim = np.zeros(100)
    test_stim[10:30] = 0.8   # 左侧干扰
    test_stim[60:80] = 1.0   # 右侧目标
    test_stim += np.random.normal(0, 0.2, 100)
    
    brain.input_stimulus(test_stim, modality=0)
    for _ in range(30):
        brain.step(dt=1.0)
    
    state = brain.get_current_state()
    print(f"  注意力焦点: 位置 {state.attention_focus} (应偏向60-80区域)")
    print(f"  注意力强度: {state.memory_stats.get('sensory_buffer_count', 0)}")
    
    return brain


def demo_dopamine_reinforcement():
    """演示4：多巴胺奖励与强化学习"""
    print("\n" + "=" * 60)
    print("演示4：多巴胺奖励与强化学习")
    print("=" * 60)
    
    brain = Brain(
        decision_neurons=4,
        action_names=["A", "B", "C", "D"]
    )
    
    print("\n--- 训练阶段：动作B总是伴随奖励 ---")
    
    # 训练：让大脑学会选择动作B
    rewards = {'A': 0.0, 'B': 1.0, 'C': 0.0, 'D': 0.0}
    
    for episode in range(20):
        # 输入刺激
        stimulus = np.random.normal(0.5, 0.3, 100)
        brain.input_stimulus(stimulus, modality=0)
        
        # 运行直到做出决策
        decision_made = False
        for step in range(50):
            state = brain.step(dt=1.0)
            if state.decision[0] >= 0 and state.decision[2] > 0.6:
                # 给予奖励
                action = state.decision[1]
                reward = rewards.get(action, 0.0)
                brain.reward(reward)
                decision_made = True
                break
        
        if episode % 5 == 0:
            state = brain.get_current_state()
            print(f"  Episode {episode}: "
                  f"决策={state.decision[1]}, "
                  f"置信度={state.decision[2]:.2f}, "
                  f"多巴胺={state.dopamine_level:.3f}")
    
    # 测试阶段
    print("\n--- 测试阶段：观察决策偏好 ---")
    
    action_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    
    for test in range(10):
        stimulus = np.random.normal(0.5, 0.3, 100)
        brain.input_stimulus(stimulus, modality=0)
        
        for step in range(50):
            state = brain.step(dt=1.0)
            if state.decision[0] >= 0 and state.decision[2] > 0.5:
                action = state.decision[1]
                if action in action_counts:
                    action_counts[action] += 1
                break
    
    print(f"  10次测试的动作选择分布：")
    for action, count in action_counts.items():
        bar = '█' * count
        print(f"    动作 {action}: {count}次 {bar}")
    
    print(f"\n  多巴胺系统价值函数条目数: {len(brain.modulation.dopamine.value_function)}")
    print(f"  决策历史长度: {len(brain.network.decision.decision_history)}")
    
    return brain


def demo_full_brain_summary():
    """演示5：完整大脑运行摘要"""
    print("\n" + "=" * 60)
    print("演示5：完整大脑运行摘要")
    print("=" * 60)
    
    brain = Brain()
    
    print("\n运行大脑 500 步，包含多种刺激...")
    
    # 模拟多种刺激和奖励
    for step in range(500):
        # 随机输入刺激
        if step % 50 == 0:
            stimulus = np.random.normal(0.5, 0.4, 100)
            reward = np.random.normal(0, 0.5)
            brain.input_stimulus(stimulus, modality=step % 5, reward=reward)
        
        brain.step(dt=1.0)
    
    # 打印摘要
    summary = brain.get_summary()
    
    print(f"\n{'='*40}")
    print("大脑运行摘要")
    print(f"{'='*40}")
    print(f"总运行时间: {summary['total_time']:.0f} ms")
    print(f"总步数: {summary['total_steps']}")
    print(f"\n神经网络:")
    print(f"  感官层神经元: {summary['sensory_neurons']}")
    print(f"  联想层神经元: {summary['association_neurons']}")
    print(f"  决策层神经元: {summary['decision_neurons']}")
    print(f"  平均感官活动: {summary['avg_sensory_activity']:.2f} Hz")
    print(f"  平均联想活动: {summary['avg_association_activity']:.2f} Hz")
    
    print(f"\n记忆系统:")
    mem = summary['memory_stats']
    print(f"  感官缓存: {mem['sensory_buffer_count']} 项")
    print(f"  短期记忆: {mem['stm_count']} 项 (平均强度: {mem['stm_avg_strength']:.3f})")
    print(f"  长期记忆: {mem['ltm_count']} 项 (平均强度: {mem['ltm_avg_strength']:.3f})")
    print(f"  语义连接: {mem['ltm_associations']} 条")
    
    print(f"\n调制系统:")
    print(f"  平均多巴胺水平: {summary['avg_dopamine']:.3f}")
    print(f"  最近情绪状态: {list(summary['recent_emotions'].keys())[:3]}")
    
    print(f"\n胶质细胞系统:")
    glia = summary['glia_cells']
    print(f"  星形胶质细胞: {glia['astrocytes']} 个")
    print(f"  少突胶质细胞: {glia['oligodendrocytes']} 个")
    print(f"  小胶质细胞: {glia['microglia']} 个")
    print(f"  星形胶质连接: {glia['astrocyte_connections']} 条")
    
    print(f"\n思考系统:")
    thought = summary['thought_system']
    ts = thought['thought_space']
    print(f"  思考空间: {ts['active_thoughts']}/{ts['capacity']} 活跃思维")
    print(f"  认知负荷: {ts['cognitive_load']:.3f}")
    print(f"  思维速度: {thought['thinking_speed']:.3f}")
    print(f"  记录思维数: {thought['thought_memory']['total_thoughts_recorded']}")
    
    print(f"\n决策统计:")
    for action, count in summary['recent_decisions'].items():
        print(f"  {action}: {count} 次")
    
    print(f"\n完整大脑对象:")
    print(brain)
    
    return brain


def main():
    """主函数：运行所有演示"""
    print("\n" + "█" * 60)
    print("█" + " " * 15 + "类脑认知架构模拟器" + " " * 18 + "█")
    print("█" * 60)
    print()
    print("架构：感官层(100) → 联想层(150) → 决策层(8)")
    print("记忆：感官缓存(15) → 短期记忆(20) → 长期记忆(500)")
    print("调制：情绪内核 ←→ 注意力调制 ←→ 多巴胺奖励")
    print("胶质：星形胶质(50) ←→ 少突胶质(20) ←→ 小胶质(10)")
    print("思考：思考空间(7) ←→ 思考记忆 ←→ 思考感官")
    print()
    
    # 运行各个演示
    brain1 = demo_basic_stimulus_response()
    brain2 = demo_memory_system()
    brain3 = demo_emotion_and_attention()
    brain4 = demo_dopamine_reinforcement()
    brain5 = demo_full_brain_summary()
    
    print("\n" + "=" * 60)
    print("所有演示完成！")
    print("=" * 60)
    print("\n提示：")
    print("  - 你可以修改参数来观察不同的大脑行为")
    print("  - 尝试不同的刺激模式和奖励方案")
    print("  - 查看源码了解每个模块的实现细节")
    print()


if __name__ == "__main__":
    main()
