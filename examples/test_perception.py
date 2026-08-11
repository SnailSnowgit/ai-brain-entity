"""
感知模块测试脚本
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain.perception import PerceptualSystem
import numpy as np

print('测试感知系统...')
perceptual = PerceptualSystem(
    visual_resolution=16,
    vocab_size=100
)

# 测试视觉
print('\n=== 视觉模块测试 ===')
test_image = np.random.rand(16, 16, 3) * 255
perceptual.visual.input_image(test_image)

for i in range(10):
    perceptual.visual.step(dt=1.0)

v_stats = perceptual.visual.get_stats()
print(f'识别物体数: {v_stats["recognized_objects"]}')
print(f'视觉记忆: {v_stats["visual_memory_items"]}')
print(f'特征活动均值: {v_stats["feature_activity_mean"]:.4f}')

# 测试语言
print('\n=== 语言模块测试 ===')
perceptual.language.input_text('the good day is here with you')

for i in range(20):
    perceptual.language.step(dt=1.0)

l_stats = perceptual.language.get_stats()
print(f'感知词汇数: {l_stats["words_perceived"]}')
print(f'理解水平: {l_stats["comprehension_level"]:.3f}')
print(f'活跃词汇: {l_stats["active_words"]}')
print(f'语音回路: {l_stats["phonological_loop_length"]} 个词')

sentence = perceptual.language.produce_sentence()
print(f'生成句子: "{sentence}"')

# 测试多模态整合
print('\n=== 多模态整合测试 ===')
perceptual.step(dt=1.0)
p_stats = perceptual.get_stats()
print(f'多模态活动均值: {p_stats["multimodal_activity_mean"]:.4f}')

print('\n✓ 感知模块测试完成！')
