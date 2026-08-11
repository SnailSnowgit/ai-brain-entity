"""验证所有架构的能力评分都≥0.98"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain.cognitive_architecture_designer import CognitiveArchitectureDesigner, TaskCategory

designer = CognitiveArchitectureDesigner()

print("=== 所有架构能力评分验证 ===")
print()

all_pass = True
for name, arch in designer.architectures.items():
    perf = designer._predict_performance(arch, TaskCategory.GENERAL_INTELLIGENCE)
    metrics = ['memory_capacity', 'processing_speed', 'stability', 
               'plasticity', 'pattern_recognition', 'emotional_sensitivity']
    
    min_score = min(perf[m] for m in metrics)
    max_score = max(perf[m] for m in metrics)
    
    status = 'PASS' if min_score >= 0.98 else 'FAIL'
    if min_score < 0.98:
        all_pass = False
    
    print(f"[{status}] {name}")
    print(f"  最低分: {min_score:.4f}  最高分: {max_score:.4f}  综合: {perf['overall']:.4f}")
    print(f"  记忆容量: {perf['memory_capacity']:.4f}")
    print(f"  处理速度: {perf['processing_speed']:.4f}")
    print(f"  稳定性:   {perf['stability']:.4f}")
    print(f"  可塑性:   {perf['plasticity']:.4f}")
    print(f"  模式识别: {perf['pattern_recognition']:.4f}")
    print(f"  情绪敏感: {perf['emotional_sensitivity']:.4f}")
    print()

print("=" * 50)
print(f"全部通过: {'YES' if all_pass else 'NO'}")
print(f"所有架构最低分 >= 0.98: {'YES' if all_pass else 'NO'}")
