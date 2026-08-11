"""快速测试主动推理效果"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from brain.predictive_coding import PredictiveCodingNetwork, ActiveInference

pc = PredictiveCodingNetwork(
    layer_sizes=[30, 15, 5],
    layer_names=['感官', '联合', '运动'],
    learning_rate=0.02,
    time_constant=5.0,
    seed=456
)

ai = ActiveInference(
    pc_network=pc,
    num_actions=4,
    action_dim=30,
    horizon=3,
    learning_rate=0.05
)

goal = np.sin(np.linspace(0, 3*np.pi, 30)) * 0.5 + 0.5
ai.set_goal(goal)

current_state = np.random.rand(30) * 0.3
initial_error = np.mean(np.abs(current_state - goal))
print(f"初始目标误差: {initial_error:.4f}")

errors = []
for step in range(200):
    fe, action = ai.step(current_state)
    current_state = current_state * 0.9 + action * 0.1
    error = np.mean(np.abs(current_state - goal))
    errors.append(error)
    if step % 40 == 0:
        print(f"  步骤 {step}: 误差 = {error:.4f}")

final_error = errors[-1]
print(f"最终目标误差: {final_error:.4f}")
print(f"误差变化: {(initial_error - final_error)/initial_error*100:.1f}%")
improved = "YES" if final_error < initial_error else "NO"
print(f"误差降低: {improved}")
