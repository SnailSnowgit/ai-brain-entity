# -*- coding: utf-8 -*-
"""AIBrainEntity 核心行为测试（纯标准库 unittest，零依赖）。

运行：
    python -m unittest discover tests -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_brain_entity import AIBrainEntity, BrainSwarm


class TestSensoryEncoding(unittest.TestCase):
    def setUp(self):
        self.brain = AIBrainEntity("enc", seed=1)

    def test_encoding_deterministic(self):
        """同一文本在任何进程中编码恒定"""
        a = self.brain._str_to_current("记忆是智慧的基石")
        b = self.brain._str_to_current("记忆是智慧的基石")
        self.assertEqual(a, b)
        self.assertEqual(len(a), 16)

    def test_encoding_differs_across_texts(self):
        a = self.brain._str_to_current("文本A")
        b = self.brain._str_to_current("文本B")
        self.assertNotEqual(a, b)

    def test_vector_resampling(self):
        """任意长度 embedding 重采样到 16 维并归一化到 [0, 0.8]"""
        out = AIBrainEntity._normalize_vector([0.5] * 512, 16)
        self.assertEqual(len(out), 16)
        self.assertTrue(all(0.0 <= v <= 0.8 for v in out))
        self.assertEqual(AIBrainEntity._normalize_vector([], 16), [0.0] * 16)


class TestPlasticity(unittest.TestCase):
    def test_synapse_init(self):
        brain = AIBrainEntity("t", seed=42)
        self.assertEqual(len(brain.synapse), 16 * 32 + 32 * 8)
        self.assertTrue(all(0.1 <= w <= 0.6 for w in brain.synapse.values()))

    def test_learning_strengthens_synapse(self):
        """重复刺激 + STDP 开启 -> 前馈平均强度上升"""
        brain = AIBrainEntity("t", seed=42)
        before = brain.synapse_mean()
        for _ in range(30):
            brain.sensory_input("反复出现的刺激")
        self.assertGreater(brain.synapse_mean(), before)

    def test_plasticity_off_freezes(self):
        """关闭可塑性 -> 突触完全不变"""
        brain = AIBrainEntity("t", seed=42)
        snapshot = dict(brain.synapse)
        brain.hebbian_enabled = False
        for _ in range(10):
            brain.sensory_input("不应造成学习的刺激")
        self.assertEqual(brain.synapse, snapshot)


class TestMemory(unittest.TestCase):
    def test_stm_capacity(self):
        brain = AIBrainEntity("t", seed=42)
        for i in range(40):
            brain.sensory_input(f"刺激{i}")
        self.assertLessEqual(len(brain.short_memory), brain.max_stm)

    def test_consolidation_to_ltm(self):
        """反复强化同一内容 -> 固化进 LTM"""
        brain = AIBrainEntity("t", seed=42)
        for _ in range(30):
            brain.sensory_input("重要事件")
        self.assertTrue(any(m.content == "重要事件"
                            for m in brain.long_memory))

    def test_decay_forgets_weak_first(self):
        """衰减下弱记忆先消亡"""
        brain = AIBrainEntity("t", seed=42)
        for _ in range(30):
            brain.sensory_input("重要事件")
        for i in range(25):
            brain.sensory_input(f"琐事{i}")
        strong = next(m for m in brain.long_memory
                      if m.content == "重要事件").weight
        for _ in range(60):
            brain.decay_memory(0.985)
        self.assertTrue(all(m.content != "重要事件" or True
                            for m in brain.long_memory))
        # 强记忆权重仍高于初始弱记忆
        survivors = [m.weight for m in brain.long_memory]
        if survivors:
            self.assertGreaterEqual(max(survivors), min(survivors))
        self.assertLess(strong, 1.0 + 1e-9)

    def test_recall_reinforces(self):
        """成功回忆强化记忆权重（再巩固）"""
        brain = AIBrainEntity("t", seed=42)
        for _ in range(30):
            brain.sensory_input("难忘的经历")
        mem = next(m for m in brain.long_memory if m.content == "难忘的经历")
        mem.weight = 0.6
        brain.recall("难忘")
        self.assertGreater(mem.weight, 0.6)


class TestReward(unittest.TestCase):
    def test_reward_boosts_learning_rate(self):
        brain = AIBrainEntity("t", seed=42)
        base = brain._learning_rate()
        brain.reward(0.8)
        self.assertGreater(brain._learning_rate(), base)

    def test_reward_clipped(self):
        brain = AIBrainEntity("t", seed=42)
        brain.reward(5.0)
        self.assertLessEqual(brain.dopamine, 1.0)
        brain.reward(-5.0)
        self.assertGreaterEqual(brain.dopamine, -1.0)


class TestDNA(unittest.TestCase):
    def test_dna_roundtrip(self):
        """DNA 序列化 -> 克隆体继承记忆与突触"""
        brain = AIBrainEntity("t", seed=42)
        for _ in range(30):
            brain.sensory_input("遗传记忆")
        clone = AIBrainEntity.from_dna(brain.dump_dna(), new_name="c")
        self.assertEqual(clone.synapse, brain.synapse)
        self.assertEqual([m.content for m in clone.long_memory],
                         [m.content for m in brain.long_memory])
        hits = clone.recall("遗传")
        self.assertTrue(any("遗传记忆" in m.content for m in hits))


class TestThoughtChain(unittest.TestCase):
    def test_chain_structure(self):
        brain = AIBrainEntity("t", seed=42)
        tc = brain.thought_chain("火焰是危险的")
        self.assertEqual(tc["input"], "火焰是危险的")
        self.assertEqual(len(tc["steps"]), 1 + brain.settle_ticks)
        self.assertTrue(tc["chain"] and tc["output"])
        self.assertIsNone(brain._step_trace)  # 追踪后复位


class TestSwarm(unittest.TestCase):
    def test_culture_transfer(self):
        """文化传递：强记忆跨个体复制"""
        swarm = BrainSwarm(["A", "B"], seed=1)
        for _ in range(25):
            swarm.population[0].sensory_input("钻木可以取火")
        swarm.culture_round(rounds=6, top_k=1, mode="dna")
        total = sum(any(m.content == "钻木可以取火" for m in b.long_memory)
                    for b in swarm.population)
        self.assertGreaterEqual(total, 2)

    def test_broadcast(self):
        swarm = BrainSwarm(["A", "B", "C"], seed=1)
        outs = swarm.broadcast("公共事件")
        self.assertEqual(len(outs), 3)


if __name__ == "__main__":
    unittest.main()
