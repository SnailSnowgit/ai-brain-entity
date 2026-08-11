"""
多智能体系统测试

测试两个智能体的交互功能
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from brain import (
    create_two_agent_system,
    AgentType,
    InteractionMode,
    MultiAgentSystem
)

print("=" * 70)
print("  多智能体系统测试")
print("  Multi-Agent System Test")
print("=" * 70)
print()

# ===== 测试1：创建两个智能体 =====
print("【测试1】创建两个智能体")
print("-" * 50)

system = create_two_agent_system(
    agent1_type=AgentType.EXPLORER,
    agent2_type=AgentType.COOPERATOR,
    interaction_mode=InteractionMode.COMMUNICATION,
    seed=42
)

print()
print(f"✓ 成功创建 {system.num_agents} 个智能体")
for agent in system.agents:
    print(f"  - {agent.name}: {agent.type.value}")
print()

# ===== 测试2：运行通信模式 =====
print("【测试2】通信模式交互")
print("-" * 50)

# 运行50步
for i in range(50):
    # 随机环境刺激
    stimulus = np.random.rand(200) * 0.3
    system.step(dt=1.0, environment_stimulus=stimulus)

summary = system.get_group_summary()
print(f"运行步数: 50")
print(f"消息总数: {summary['total_messages']}")
print(f"交互总数: {summary['total_interactions']}")
print()

# ===== 测试3：观察模式 =====
print("【测试3】观察模式交互")
print("-" * 50)

system2 = create_two_agent_system(
    agent1_type=AgentType.IMITATOR,
    agent2_type=AgentType.LEADER,
    interaction_mode=InteractionMode.OBSERVATION,
    seed=123
)

for i in range(30):
    stimulus = np.random.rand(200) * 0.3
    system2.step(dt=1.0, environment_stimulus=stimulus)

summary2 = system2.get_group_summary()
print(f"运行步数: 30")
print(f"交互总数: {summary2['total_interactions']}")
print()

# 检查社会记忆
agent0 = system2.agents[0]
print(f"{agent0.name} 的社会记忆:")
for aid, memory in agent0.social_memory.items():
    print(f"  - {memory.agent_name}: "
          f"交互{memory.interaction_count}次, "
          f"信任度{memory.trust_level:.3f}, "
          f"相似度{memory.similarity:.3f}")
print()

# ===== 测试4：合作模式 =====
print("【测试4】合作模式")
print("-" * 50)

system3 = create_two_agent_system(
    agent1_type=AgentType.COOPERATOR,
    agent2_type=AgentType.COOPERATOR,
    interaction_mode=InteractionMode.COOPERATION,
    seed=456
)

# 让两个智能体靠近
system3.agents[0].position = np.array([50, 50])
system3.agents[1].position = np.array([55, 50])

for i in range(20):
    stimulus = np.random.rand(200) * 0.2
    system3.step(dt=1.0, environment_stimulus=stimulus)

summary3 = system3.get_group_summary()
print(f"运行步数: 20")
print(f"合作事件: {summary3['cooperation_events']}")
print(f"竞争事件: {summary3['competition_events']}")
print()

# ===== 测试5：竞争模式 =====
print("【测试5】竞争模式")
print("-" * 50)

system4 = create_two_agent_system(
    agent1_type=AgentType.COMPETITOR,
    agent2_type=AgentType.COMPETITOR,
    interaction_mode=InteractionMode.COMPETITION,
    seed=789
)

# 让两个智能体靠近
system4.agents[0].position = np.array([50, 50])
system4.agents[1].position = np.array([52, 50])

for i in range(20):
    stimulus = np.random.rand(200) * 0.2
    system4.step(dt=1.0, environment_stimulus=stimulus)

summary4 = system4.get_group_summary()
print(f"运行步数: 20")
print(f"合作事件: {summary4['cooperation_events']}")
print(f"竞争事件: {summary4['competition_events']}")
print()

# ===== 测试6：系统状态打印 =====
print("【测试6】系统状态")
print("-" * 50)
system.print_status()
print()

print("=" * 70)
print("  ✓ 多智能体系统测试完成！")
print("=" * 70)
