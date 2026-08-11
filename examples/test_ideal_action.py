"""测试：如果动作完全指向目标，误差会怎样？"""
import numpy as np

goal = np.sin(np.linspace(0, 3*np.pi, 30)) * 0.5 + 0.5
current_state = np.random.RandomState(42).rand(30) * 0.3

initial_error = np.mean(np.abs(current_state - goal))
print(f"初始目标误差: {initial_error:.4f}")

errors = []
for step in range(200):
    # 动作 = 目标方向（单位向量）
    direction = goal - current_state
    norm = np.linalg.norm(direction)
    if norm > 0:
        direction /= norm
    
    # 状态更新
    current_state = current_state * 0.9 + direction * 0.1
    
    error = np.mean(np.abs(current_state - goal))
    errors.append(error)
    
    if step % 40 == 0:
        print(f"  步骤 {step}: 误差 = {error:.4f}")

final_error = errors[-1]
print(f"最终目标误差: {final_error:.4f}")
print(f"误差变化: {(initial_error - final_error)/initial_error*100:.1f}%")
