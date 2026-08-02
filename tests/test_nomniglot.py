# -*- coding: utf-8 -*-
"""N-Omniglot 数据接入测试（纯标准库 unittest）。"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_brain_entity import AIBrainEntity
from nomniglot import (classes_of, few_shot_split, load,
                       nearest_prototype_accuracy, sample_to_vector, stats)

DATA = "data/nomniglot_latin.json"


class TestLoading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load(DATA)
        cls.samples = cls.data["samples"]

    def test_structure(self):
        self.assertEqual(len(self.samples), 130)          # 13 类 × 10 样本
        self.assertEqual(len(classes_of(self.samples)), 13)
        s = self.samples[0]
        self.assertEqual(len(s["frames"]), 4)             # 4 帧
        self.assertEqual(len(s["frames"][0]), 16)         # 16×16 网格
        self.assertGreater(s["n_events"], 1000)           # 真实事件数据非空

    def test_value_range(self):
        for s in self.samples[:10]:
            for f in s["frames"]:
                for row in f:
                    for v in row:
                        self.assertGreaterEqual(v, -1.0)
                        self.assertLessEqual(v, 1.0)
        # 每帧 max-normalize：至少有一个 |v| == 1（非空帧）
        s = self.samples[0]
        self.assertTrue(any(abs(v) == 1.0 for f in s["frames"]
                            for row in f for v in row))

    def test_vector(self):
        v = sample_to_vector(self.samples[0])
        self.assertEqual(len(v), 256)
        vf = sample_to_vector(self.samples[0], mode="flatten")
        self.assertEqual(len(vf), 4 * 256)

    def test_stats(self):
        st = stats(self.samples)
        self.assertEqual(st["n_samples"], 130)
        self.assertEqual(st["n_classes"], 13)
        self.assertGreater(st["events_mean"], 10000)


class TestFewShot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples = load(DATA)["samples"]

    def test_split_deterministic(self):
        a = few_shot_split(self.samples, 3, seed=7)
        b = few_shot_split(self.samples, 3, seed=7)
        self.assertEqual([s["sample"] for s in a[0]],
                         [s["sample"] for s in b[0]])
        self.assertEqual(len(a[0]), 13 * 3)
        self.assertEqual(len(a[1]), 130 - 13 * 3)

    def test_baseline_above_chance(self):
        sup, qry = few_shot_split(self.samples, 5, seed=7)
        acc = nearest_prototype_accuracy(sup, qry)
        self.assertGreater(acc, 1 / 13)   # 显著高于随机
        self.assertLess(acc, 1.0)         # 且不是平凡满分（数据确有挑战）


class TestBrainIntegration(unittest.TestCase):
    def test_vector_drives_brain(self):
        samples = load(DATA)["samples"]
        brain = AIBrainEntity("t_nom", seed=1)
        tick0 = brain.tick
        for s in samples[:26]:
            brain.sensory_input_vector(
                sample_to_vector(s), label=f"拉丁字符{s['class']:02d}")
            brain.reward(0.6)
        self.assertEqual(brain.tick, tick0 + 26)
        self.assertGreaterEqual(len(brain.long_memory), 1)  # 有记忆固化
        hits = brain.recall("拉丁字符")
        self.assertTrue(hits)

    def test_novelty_habituation(self):
        # v4.9.1 习惯化：同一内容反复暴露，新奇度逐次折扣衰减
        samples = load(DATA)["samples"]
        brain = AIBrainEntity("t_nov", seed=2)
        vec = sample_to_vector(samples[0])
        novs = []
        for _ in range(4):
            brain.sensory_input_vector(vec, label="已学字符")
            novs.append(brain.novelty)
        self.assertGreater(novs[0], 0.3)              # 首次：新奇
        self.assertLess(novs[-1], novs[0])            # 重复暴露后衰减
        self.assertTrue(all(a >= b for a, b in zip(novs, novs[1:])))


if __name__ == "__main__":
    unittest.main()
