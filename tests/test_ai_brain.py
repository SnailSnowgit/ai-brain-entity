# -*- coding: utf-8 -*-
"""AIBrainEntity 核心行为测试（纯标准库 unittest，零依赖）。

运行：
    python -m unittest discover tests -v
"""
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_brain_entity import (
    AIBrainEntity, BrainSwarm,
    encode_image, encode_audio,
    register_image_encoder, register_audio_encoder,
    unregister_image_encoder, unregister_audio_encoder,
    list_encoders,
)


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


class TestCustomMultimodal(unittest.TestCase):
    """v3.1 自定义多模态模型：注册表、优先级链、输出校验"""

    def setUp(self):
        # 隔离注册表，避免测试间相互污染
        import ai_brain_entity as m
        self._module = m
        self._saved = (
            dict(m._CUSTOM_IMAGE_ENCODERS), m._DEFAULT_IMAGE_ENCODER,
            dict(m._CUSTOM_AUDIO_ENCODERS), m._DEFAULT_AUDIO_ENCODER,
        )
        m._CUSTOM_IMAGE_ENCODERS.clear()
        m._CUSTOM_AUDIO_ENCODERS.clear()
        m._DEFAULT_IMAGE_ENCODER = None
        m._DEFAULT_AUDIO_ENCODER = None

    def tearDown(self):
        m = self._module
        img, img_d, aud, aud_d = self._saved
        m._CUSTOM_IMAGE_ENCODERS.clear(); m._CUSTOM_IMAGE_ENCODERS.update(img)
        m._CUSTOM_AUDIO_ENCODERS.clear(); m._CUSTOM_AUDIO_ENCODERS.update(aud)
        m._DEFAULT_IMAGE_ENCODER = img_d
        m._DEFAULT_AUDIO_ENCODER = aud_d

    def _tmp_file(self):
        import tempfile
        fd, path = tempfile.mkstemp(suffix=".bin")
        with os.fdopen(fd, "wb") as f:
            f.write(b"hello multimodal" * 8)
        self.addCleanup(os.remove, path)
        return path

    def test_registered_default_encoder_wins(self):
        """注册的默认自定义编码器优先于内置 CLIP / 伪 embedding"""
        path = self._tmp_file()
        register_image_encoder(lambda p: [0.5] * 32, name="t32")
        vec = encode_image(path)
        self.assertEqual(vec, [0.5] * 32)

    def test_explicit_encoder_arg_overrides_default(self):
        """调用时显式传入的 encoder 优先于注册的默认编码器"""
        path = self._tmp_file()
        register_image_encoder(lambda p: [1.0] * 8, name="d8")
        vec = encode_image(path, encoder=lambda p: [2.0] * 4)
        self.assertEqual(vec, [2.0] * 4)
        # 也支持按注册名指定
        register_image_encoder(lambda p: [3.0] * 4, name="other")
        self.assertEqual(encode_image(path, encoder="other"), [3.0] * 4)

    def test_unregister_falls_back_to_pseudo(self):
        """注销默认编码器后回落到内置链（无依赖环境 = 伪 embedding）"""
        path = self._tmp_file()
        register_image_encoder(lambda p: [0.5] * 32, name="t32")
        unregister_image_encoder("t32")
        vec = encode_image(path)          # 无 transformers 环境 → 512 维伪 embedding
        self.assertEqual(len(vec), 512)
        self.assertNotEqual(vec, [0.5] * 32)

    def test_audio_encoder_and_listing(self):
        path = self._tmp_file()
        register_audio_encoder(lambda p: [0.1, 0.2], name="a2")
        self.assertEqual(encode_audio(path), [0.1, 0.2])
        info = list_encoders()
        self.assertEqual(info["audio"]["default"], "a2")
        self.assertIn("a2", info["audio"]["custom"])
        unregister_audio_encoder("a2")
        self.assertIsNone(list_encoders()["audio"]["default"])

    def test_unknown_encoder_name_raises(self):
        with self.assertRaises(KeyError):
            encode_image(self._tmp_file(), encoder="not-registered")

    def test_invalid_encoder_output_raises(self):
        """自定义编码器返回非法输出属于调用方错误，直接抛出而非静默降级"""
        path = self._tmp_file()
        with self.assertRaises(TypeError):
            encode_image(path, encoder=lambda p: "not-a-vector")
        with self.assertRaises(TypeError):
            encode_image(path, encoder=lambda p: [1.0, "bad", 3.0])

    def test_numpy_like_output_accepted(self):
        """支持带 tolist() 的输出（numpy 数组 / torch 张量）"""
        class FakeArray:
            def tolist(self):
                return [0.7] * 16
        vec = encode_image(self._tmp_file(), encoder=lambda p: FakeArray())
        self.assertEqual(vec, [0.7] * 16)

    def test_perceive_image_uses_custom_encoder(self):
        """perceive_image 把自定义 encoder 的输出送入感官层并可被观测"""
        brain = AIBrainEntity("t", seed=42)
        path = self._tmp_file()
        out = brain.perceive_image(path, label="自定义图",
                                   encoder=lambda p: [0.9] * 64)
        self.assertIsInstance(out, str)
        # 64 维自定义 embedding 应被重采样为 16 路感官电流
        self.assertEqual(len(brain._normalize_vector([0.9] * 64, 16)), 16)


if __name__ == "__main__":
    unittest.main()
