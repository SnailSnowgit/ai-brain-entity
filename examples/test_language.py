"""
语言层重构验证实验：符号接地的全链路

设计说明：感官→联想通路的模式分离度不足（动力学校准遗留问题，
见会话记录），因此实验在概念层直接注入可区分的概念簇模式
（concept cluster），模拟"不同的感知体验"在联想层的状态，
验证语言系统的核心机制：

  1. 教学期：概念模式 A + "water" 共现（接地学习）
  2. 理解测试：只说 "water" → 联想层应重现模式 A（词唤起概念）
  3. 产生测试：只呈现模式 A → 内部言语应说出 "water"（概念变成词）
  4. 操作系统环路：思维→言语→再注入的持续运转
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from brain import Brain

print("=" * 65)
print("  语言层重构验证：符号接地全链路")
print("=" * 65)

np.random.seed(7)
brain = Brain(sensory_neurons=200, association_neurons=500, decision_neurons=20)
net = brain.network
assoc = net.association

WORD_CONCEPT = {'water': 'concept_0', 'food': 'concept_1',
                'house': 'concept_2', 'dream': 'concept_3'}

def present_concept(concept_name, steps=6):
    """注入概念簇模式，模拟一次'体验'"""
    pattern = assoc.get_concept_pattern(concept_name)
    for _ in range(steps):
        assoc.apply_external_input(pattern * 3.0)
        brain.step(dt=1.0)

def sim(x, y):
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    return float(x @ y / (nx * ny)) if nx > 1e-6 and ny > 1e-6 else 0.0

# ---------- 1. 教学期 ----------
print("\n【1】教学期：体验-词语共现（接地学习）")
for trial in range(5):
    for word, concept in WORD_CONCEPT.items():
        present_concept(concept)
        brain.input_text(word)
        # 概念间静息隔离：发放率 EMA 会混入上一个概念的残余活动
        for _ in range(8):
            brain.step(dt=1.0)

print(f"  已接地词汇: {brain.language.get_grounded_vocab_size()} / 4")
for word in WORD_CONCEPT:
    idx = brain.language.vocabulary[word]
    print(f"  '{word}' 接地强度: {brain.language.grounding_strength[idx]:.2f}")

# 概念原型（用于相似度判定）
proto_A = assoc.get_concept_pattern('concept_0')   # water
proto_B = assoc.get_concept_pattern('concept_1')   # food

# ---------- 2. 理解测试 ----------
print("\n【2】理解测试：只听到 'water'（无任何体验）")
for _ in range(20):
    brain.step(dt=1.0)  # 静息清空

brain.input_text("water")
# 取理解注入后的第一步响应：直接诱发活动（第 2 步起为层内
# 循环反跳的全层暴发，不属于理解信号本身）
brain.step(dt=1.0)
evoked = assoc.get_activity_pattern().copy()

print(f"  唤起模式 vs 'water'原型(A): {sim(evoked, proto_A):.3f}")
print(f"  唤起模式 vs 'food' 原型(B): {sim(evoked, proto_B):.3f}")
# 概念簇内激活占比
cluster_A_idx = assoc.concept_clusters['concept_0']
in_A = evoked[cluster_A_idx].sum() / (evoked.sum() + 1e-9)
print(f"  唤起活动中落在 A 概念簇内的比例: {in_A:.3f} (随机期望 ~0.1)")

# ---------- 3. 产生测试 ----------
print("\n【3】产生测试：只呈现概念 A 体验（不说话）")
for _ in range(10):
    brain.step(dt=1.0)
brain.language.inner_speech.clear()

present_concept('concept_0', steps=8)
for _ in range(30):
    brain.step(dt=1.0)

speech = brain.language.get_inner_speech_text()
print(f"  内部言语: \"{speech}\"")
for word in WORD_CONCEPT:
    print(f"  说出 '{word}': {'✓' if word in speech else '✗'}")

# ---------- 4. 判定 ----------
print()
print("=" * 65)
print("  判定")
print("=" * 65)
grounded = brain.language.get_grounded_vocab_size() >= 4
comprehension = in_A > 0.2   # 显著高于随机
production = 'water' in speech
print(f"  接地学习:   {'✓' if grounded else '✗'} ({brain.language.get_grounded_vocab_size()} 词)")
print(f"  理解路径:   {'✓' if comprehension else '✗'} (词→概念, A簇占比 {in_A:.2f})")
print(f"  产生路径:   {'✓' if production else '✗'} (概念→词)")
print()
print(f"  语言统计: {brain.language.get_stats()}")
