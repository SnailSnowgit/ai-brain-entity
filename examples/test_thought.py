"""
思考系统测试
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain import (
    ThoughtSpace, ThoughtMemory, ThoughtSensory, ThoughtSystem, 
    Thought, ThoughtType, Brain
)

print("=" * 60)
print("思考系统测试")
print("=" * 60)

print("\n测试1：思考空间 (ThoughtSpace)")
print("-" * 40)

ts = ThoughtSpace(capacity=5, vector_dim=10)
print(f"思考空间容量: {ts.capacity}")

# 提交几个思维内容
for i in range(8):
    content = np.random.randn(10)
    content = content / np.linalg.norm(content)
    thought = Thought(
        content=content,
        thought_type=ThoughtType.PERCEPTUAL if i % 2 == 0 else ThoughtType.MEMORY,
        strength=0.3 + i * 0.1,
        source=f"source_{i}"
    )
    ts.submit_thought(thought)

print(f"候选思维数: {len(ts.candidate_thoughts)}")

# 选择进入意识
print("\n选择思维进入意识...")
for i in range(5):
    selected = ts.select_for_consciousness()
    if selected:
        print(f"  第{i}次: 类型={selected.thought_type.value}, "
              f"强度={selected.strength:.2f}, 来源={selected.source}")

print(f"\n活跃思维数: {len(ts.active_thoughts)}")
print(f"认知负荷: {ts.cognitive_load:.3f}")
print(f"当前广播内容类型: {ts.broadcast_content.thought_type.value}")

print("\n测试2：思考记忆 (ThoughtMemory)")
print("-" * 40)

tm = ThoughtMemory(max_episodes=10, max_thoughts_per_episode=20)
print(f"最大情节数: {tm.max_episodes}")

# 记录一些思维
print("\n记录思维序列...")
for i in range(15):
    content = np.random.randn(10)
    thought = Thought(
        content=content,
        thought_type=ThoughtType.ABSTRACT,
        strength=0.5 + np.random.random() * 0.3,
        duration=np.random.random() * 5
    )
    tm.record_thought(thought)

print(f"当前序列长度: {len(tm.current_sequence)}")
print(f"总思维数: {tm.thought_patterns['total_thoughts']}")
print(f"类型分布: {tm.thought_patterns['type_distribution']}")

# 结束一个情节
tm._end_episode()
print(f"\n结束一个情节后:")
print(f"  存储的情节数: {len(tm.thought_episodes)}")
print(f"  当前序列: {len(tm.current_sequence)}")

# 测试元记忆
print("\n元记忆:")
for key, value in tm.metamemory.items():
    print(f"  {key}: {value:.2f}")

# 更新元记忆
tm.update_metamemory({'thinking_speed': 0.8, 'memory_confidence': 0.9})
print("\n更新后元记忆:")
for key, value in tm.metamemory.items():
    print(f"  {key}: {value:.2f}")

print("\n测试3：思考感官 (ThoughtSensory)")
print("-" * 40)

ts_sensory = ThoughtSensory()
print("初始认知状态:")
for key, value in ts_sensory.cognitive_state.items():
    print(f"  {key}: {value:.3f}")

# 感知一个思维
print("\n感知一个高强度思维...")
thought = Thought(
    content=np.random.randn(10),
    thought_type=ThoughtType.PERCEPTUAL,
    strength=0.8
)
ts_sensory.perceive_thought(thought, cognitive_load=0.5, memory_strength=0.6)

print("感知后认知状态:")
for key, value in ts_sensory.cognitive_state.items():
    print(f"  {key}: {value:.3f}")

# 测试冲突检测
print("\n检测思维冲突...")
thought1 = Thought(content=np.array([1.0, 0.0, 0.0]), thought_type=ThoughtType.ABSTRACT)
thought2 = Thought(content=np.array([-1.0, 0.0, 0.0]), thought_type=ThoughtType.ABSTRACT)
conflict = ts_sensory.detect_conflict(thought1, thought2)
print(f"  冲突程度: {conflict:.3f}")

# 测试顿悟
print("\n触发顿悟时刻...")
ts_sensory.trigger_aha_moment()
print(f"  顿悟信号: {ts_sensory.feedback_signals['aha_moment']:.3f}")
print(f"  流畅度: {ts_sensory.cognitive_state['fluency']:.3f}")
print(f"  自信度: {ts_sensory.cognitive_state['confidence']:.3f}")

# 测试内省
print("\n开始内省...")
ts_sensory.start_introspection(depth=0.7)
print(f"  内省激活: {ts_sensory.introspection_active}")
print(f"  内省深度: {ts_sensory.introspection_depth}")
print(f"  认知负荷: {ts_sensory.cognitive_state['cognitive_load']:.3f}")

print("\n测试4：思考系统整合")
print("-" * 40)

thought_sys = ThoughtSystem(
    thought_space_capacity=7,
    thought_vector_dim=50,
    max_thought_episodes=50
)

print("思考系统创建成功")
summary = thought_sys.get_summary()
print(f"  思考空间容量: {summary['thought_space']['capacity']}")
print(f"  思维速度: {summary['thinking_speed']:.2f}")

# 输入各种思维
print("\n输入各种思维内容...")
for i in range(20):
    if i % 3 == 0:
        # 感知性思维
        content = np.random.randn(50)
        content = content / np.linalg.norm(content)
        thought_sys.input_perceptual_thought(content, strength=0.4 + np.random.random() * 0.3)
    elif i % 3 == 1:
        # 记忆性思维
        content = np.random.randn(50)
        content = content / np.linalg.norm(content)
        thought_sys.input_memory_thought(content, strength=0.3 + np.random.random() * 0.2)
    else:
        # 情绪性思维
        valence = np.random.uniform(-1, 1)
        arousal = np.random.uniform(0, 1)
        thought_sys.input_emotional_thought(valence, arousal, strength=0.5)

# 运行几步
print("\n运行思考系统...")
for i in range(30):
    state = thought_sys.step(dt=1.0)
    if i % 10 == 0:
        print(f"  第{i}步:")
        print(f"    活跃思维: {state['space']['active_thoughts']}")
        print(f"    认知负荷: {state['space']['cognitive_load']:.3f}")
        print(f"    思维速度: {state['thinking_speed']:.3f}")
        if state['space']['broadcast_type']:
            print(f"    当前思维类型: {state['space']['broadcast_type']}")

# 获取思维流
stream = thought_sys.get_thought_stream(last_n=5)
print(f"\n最近5个思维:")
for i, t in enumerate(stream):
    print(f"  {i+1}. {t.thought_type.value} (强度={t.strength:.2f})")

# 摘要
final_summary = thought_sys.get_summary()
print(f"\n思考系统摘要:")
print(f"  处理的总思维数: {final_summary['total_thoughts_processed']}")
print(f"  存储的情节数: {final_summary['thought_memory']['episodes_stored']}")
print(f"  认知状态-流畅度: {final_summary['thought_sensory']['cognitive_state']['fluency']:.3f}")
print(f"  认知状态-自信度: {final_summary['thought_sensory']['cognitive_state']['confidence']:.3f}")

print("\n测试5：完整大脑中的思考系统")
print("-" * 40)

brain = Brain(
    sensory_neurons=50,
    association_neurons=80,
    decision_neurons=4
)

print("创建大脑成功")
print(f"思考空间容量: {brain.thought.space.capacity}")

# 运行几步
print("\n运行大脑，观察思考系统...")
for i in range(30):
    stimulus = np.random.rand(50) * 2.0
    brain.input_stimulus(stimulus, modality=0)
    state = brain.step(dt=1.0)
    if i % 10 == 0:
        print(f"  第{i}步:")
        print(f"    感官活动: {state.sensory_activity:.1f} Hz")
        print(f"    思考-活跃思维: {state.thought_stats['space']['active_thoughts']}")
        print(f"    思考-认知负荷: {state.thought_stats['space']['cognitive_load']:.3f}")
        print(f"    思考-流畅度: {state.thought_stats['sensory']['cognitive_state']['fluency']:.3f}")

# 完整摘要
brain_summary = brain.get_summary()
print(f"\n大脑思考系统摘要:")
ts_info = brain_summary['thought_system']
print(f"  思考空间: {ts_info['thought_space']['active_thoughts']}活跃 / {ts_info['thought_space']['capacity']}容量")
print(f"  思维速度: {ts_info['thinking_speed']:.3f}")
print(f"  记录思维数: {ts_info['thought_memory']['total_thoughts_recorded']}")

print("\n" + "=" * 60)
print("所有测试完成！")
print("=" * 60)
