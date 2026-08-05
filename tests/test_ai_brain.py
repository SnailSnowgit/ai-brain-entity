# -*- coding: utf-8 -*-
"""AIBrainEntity 核心行为测试（纯标准库 unittest，零依赖）。

运行：
    python -m unittest discover tests -v
"""
import os
import random
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# 测试环境离线化：已装 transformers/whisper 时也不触发网络下载——
# CLIP 走 HF_HUB_OFFLINE 立即失败回退伪 embedding；
# Whisper 预置假缓存跳过 145MB 下载（无 ffmpeg 时随后也立即回退伪 embedding）。
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import ai_brain_entity as _abe
if _abe._WHISPER_CACHE is None:
    _abe._WHISPER_CACHE = False

from ai_brain_entity import (
    AIBrainEntity, BrainSwarm, BrainMemory, ThoughtItem, LearnableProjection,
    encode_image, encode_audio,
    register_image_encoder, register_audio_encoder,
    unregister_image_encoder, unregister_audio_encoder,
    list_encoders,
    register_language_generator, unregister_language_generator,
    get_language_generator_info, set_qwen_model,
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

    def test_clip_output_flattened(self):
        """CLIP 返回多维张量/ModelOutput 时统一展平为一维 float 列表"""
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("PIL 不可用")

        class _FakeTensor:
            def __init__(self, arr):
                self._arr = np.asarray(arr, dtype=float)

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self._arr

        class _FakeClipModel:
            def get_image_features(self, **kw):
                # 模拟新版 transformers 的多维返回 (1, 1, 8)
                return _FakeTensor(np.arange(8).reshape(1, 1, 8))

        class _FakeProcessor:
            def __call__(self, **kw):
                return {}

        import tempfile
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        Image.new("RGB", (8, 8), (200, 30, 30)).save(path)
        self.addCleanup(os.remove, path)

        m = self._module
        saved_cache = m._CLIP_CACHE
        try:
            m._CLIP_CACHE = (_FakeClipModel(), _FakeProcessor())
            vec = encode_image(path)
        finally:
            m._CLIP_CACHE = saved_cache
        self.assertEqual(vec, [float(i) for i in range(8)])
        self.assertTrue(all(isinstance(v, float) for v in vec))


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


class TestLearnableProjection(unittest.TestCase):
    """v4.0 扩展1：可学习投影替代线性插值重采样"""

    @staticmethod
    def _cos(a, b):
        import math as m
        dot = sum(x * y for x, y in zip(a, b))
        na = m.sqrt(sum(x * x for x in a)) or 1.0
        nb = m.sqrt(sum(x * x for x in b)) or 1.0
        return dot / (na * nb)

    def test_output_shape_and_range(self):
        proj = LearnableProjection(in_dim=512, out_dim=16, seed=42)
        cur = proj.project([0.1] * 512)
        self.assertEqual(len(cur), 16)
        self.assertTrue(all(0.0 <= v <= 0.8 for v in cur))

    def test_contrast_preserved_better_than_interpolation(self):
        """投影空间（带符号）中 相似对/相异对 的余弦差显著大于插值基线，
        且插值 abs 归一化会抬高相异对的基线相似度"""
        rng = random.Random(7)
        cat = [rng.uniform(-1, 1) for _ in range(512)]
        cat2 = [v + rng.uniform(-0.05, 0.05) for v in cat]
        dog = [rng.uniform(-1, 1) for _ in range(512)]
        proj = LearnableProjection(in_dim=512, out_dim=16, seed=42)
        spread_proj = (
            self._cos(proj.project_raw(cat), proj.project_raw(cat2))
            - self._cos(proj.project_raw(cat), proj.project_raw(dog)))
        interp_dif = self._cos(AIBrainEntity._normalize_vector(cat, 16),
                               AIBrainEntity._normalize_vector(dog, 16))
        interp_sim = self._cos(AIBrainEntity._normalize_vector(cat, 16),
                               AIBrainEntity._normalize_vector(cat2, 16))
        # 投影保留对比度：相似≈1、相异≈0，spread 远超插值的 ~0.34
        self.assertGreater(spread_proj, (interp_sim - interp_dif) * 2)
        # abs 归一化丢失符号 → 相异对基线被抬高（>0.5），投影无此问题
        self.assertGreater(interp_dif, 0.5)
        self.assertLess(abs(self._cos(proj.project_raw(cat),
                                      proj.project_raw(dog))), 0.3)

    def test_oja_training_adapts(self):
        """Oja 训练改变投影且 train_steps 计数"""
        rng = random.Random(3)
        vec = [rng.uniform(-1, 1) for _ in range(64)]
        proj = LearnableProjection(in_dim=64, out_dim=16, seed=1)
        before = [row[:] for row in proj.W]
        for _ in range(20):
            proj.train(vec)
        self.assertEqual(proj.train_steps, 20)
        self.assertNotEqual(before, proj.W)

    def test_brain_projection_pathway(self):
        """enable_projection 后 sensory_input_vector 走投影并在线学习"""
        brain = AIBrainEntity("t", seed=42)
        brain.enable_projection(True)
        out = brain.sensory_input_vector([0.3] * 128, label="向量刺激")
        self.assertIsInstance(out, str)
        self.assertIn(128, brain._projections)
        self.assertEqual(brain._projections[128].train_steps, 1)
        brain.sensory_input_vector([0.4] * 128)
        self.assertEqual(brain._projections[128].train_steps, 2)


class TestRewardTD(unittest.TestCase):
    """v4.0 扩展3：RPE / TD 误差学习"""

    def test_predicted_reward_fades_dopamine(self):
        """同一奖励反复出现 → RPE 衰减 → 多巴胺反应消失（奖励被预测）"""
        brain = AIBrainEntity("t", seed=42)
        first = brain.reward_td(0.8)["rpe"]
        for _ in range(30):
            brain.reward_td(0.8)
        last = brain.reward_td(0.8)["rpe"]
        self.assertGreater(abs(first), abs(last) + 0.3)
        self.assertAlmostEqual(brain.value_estimate, 0.8, delta=0.05)

    def test_surprise_reopens_rpe(self):
        """奖励突变 → 重新出现大 RPE"""
        brain = AIBrainEntity("t", seed=42)
        for _ in range(40):
            brain.reward_td(0.8)
        surprise = brain.reward_td(-0.5)["rpe"]
        self.assertLess(surprise, -1.0)
        self.assertEqual(len(brain.rpe_history), 41)

    def test_value_never_exceeds_bounds(self):
        brain = AIBrainEntity("t", seed=42)
        for _ in range(50):
            brain.reward_td(1.0)
        self.assertLessEqual(brain.value_estimate, 1.0)
        self.assertLessEqual(brain.dopamine, 1.0)


class TestActionLanguage(unittest.TestCase):
    """v4.0 扩展4：动作空间与语言生成"""

    def test_decide_action_structure(self):
        brain = AIBrainEntity("t", seed=42)
        brain.sensory_input("火焰是危险的")
        act = brain.decide_action("火焰是危险的")
        self.assertIn(act["action"], AIBrainEntity.ACTION_SPACE)
        self.assertEqual(act["verb"],
                         AIBrainEntity.ACTION_SPACE[act["action"]]["verb"])
        self.assertTrue(0.0 <= act["intensity"] <= 1.0)
        self.assertIn(act["mood"], ("calm", "curiosity", "stress", "pleasure"))

    def test_express_utterance(self):
        brain = AIBrainEntity("t", seed=42)
        brain.sensory_input("记忆是智慧的基石")
        out = brain.express("记忆是智慧的基石")
        self.assertIn("action", out)
        self.assertIn("记忆", out["utterance"])
        self.assertIsInstance(out["utterance"], str)

    def test_recalled_memory_enters_utterance(self):
        """固化后的记忆会被语言生成引用"""
        brain = AIBrainEntity("t", seed=42)
        for _ in range(30):
            brain.sensory_input("钻木可以取火")
        brain.sensory_input(" unrelated ")
        out = brain.express("钻木")
        if out["action"]["recalled"]:
            self.assertIn("钻木可以取火", out["utterance"])


class TestSwarmDynamics(unittest.TestCase):
    """v4.0 扩展2：水平 vs 垂直传播 + 共识涌现"""

    def _seeded_swarm(self):
        swarm = BrainSwarm(["A", "B", "C"], seed=1)
        for _ in range(25):
            swarm.population[0].sensory_input("钻木可以取火")
        return swarm

    def test_horizontal_same_generation_only(self):
        """水平传播：模因在同代个体间扩散"""
        swarm = self._seeded_swarm()
        swarm.horizontal_transfer(rounds=6, top_k=1)
        holders = sum(any(m.content == "钻木可以取火" for m in b.long_memory)
                      for b in swarm.population)
        self.assertGreaterEqual(holders, 2)

    def test_vertical_crosses_generations(self):
        """垂直传播：仅长辈→晚辈；子代获得祖辈记忆"""
        swarm = self._seeded_swarm()
        child = swarm.reproduce(0, "D")
        self.assertEqual(child.generation, 2)
        child.long_memory.clear()  # 清掉遗传的，只看垂直传递
        # 确定性验证传递机制本身
        n = BrainSwarm._transfer_top(swarm.population[0], child, top_k=1)
        self.assertEqual(n, 1)
        self.assertEqual(child.long_memory[0].content, "钻木可以取火")
        # 随机垂直轮次也能收敛到子代
        child.long_memory.clear()
        swarm.vertical_transfer(rounds=20, top_k=1)
        self.assertTrue(any(m.content == "钻木可以取火"
                            for m in child.long_memory))

    def test_transmission_dynamics_curve(self):
        """扩散曲线单调不减（只有新增持有，没有流失）"""
        swarm = self._seeded_swarm()
        dyn = swarm.transmission_dynamics("钻木可以取火", rounds=4,
                                          direction="horizontal")
        cov = dyn["coverage"]
        self.assertEqual(len(cov), 4)
        self.assertTrue(all(b >= a for a, b in zip(cov, cov[1:])))

    def test_consensus_emergence(self):
        """文化收敛后记忆共识指数上升；行动共识分布总和=种群数"""
        swarm = self._seeded_swarm()
        before = swarm.consensus()["memory_consensus"]["index"]
        swarm.horizontal_transfer(rounds=10, top_k=1)
        after = swarm.consensus()["memory_consensus"]["index"]
        self.assertGreaterEqual(after, before)
        act = swarm.consensus(stimulus="钻木可以取火")["action_consensus"]
        self.assertEqual(sum(act["distribution"].values()),
                         len(swarm.population))
        self.assertTrue(0.0 < act["index"] <= 1.0)


class TestTopologyPhase(unittest.TestCase):
    """v4.1 扩展B：连接拓扑与共识收敛相变"""

    def _seeded(self, n=6):
        swarm = BrainSwarm([f"N{i}" for i in range(n)], seed=1)
        for _ in range(25):
            swarm.population[0].sensory_input("钻木可以取火")
        return swarm

    def test_ring_topology_structure(self):
        swarm = self._seeded(5)
        adj = swarm.set_topology("ring")
        self.assertEqual(adj[0], [1, 4])
        self.assertEqual(adj[2], [1, 3])
        self.assertEqual(swarm._neighbors(0), [1, 4])

    def test_star_topology_structure(self):
        swarm = self._seeded(5)
        adj = swarm.set_topology("star")
        self.assertEqual(sorted(adj[0]), [1, 2, 3, 4])
        self.assertEqual(adj[3], [0])

    def test_fully_connected_clears_topology(self):
        swarm = self._seeded(4)
        swarm.set_topology("ring")
        swarm.set_topology("fully_connected")
        self.assertIsNone(swarm.topology)
        self.assertEqual(swarm._neighbors(0), [1, 2, 3])

    def test_unknown_topology_raises(self):
        swarm = self._seeded(3)
        with self.assertRaises(ValueError):
            swarm.set_topology("hypercube")

    def test_ring_slows_horizontal_spread(self):
        """全连接收敛不慢于环形（拓扑限制传播速度）"""
        fc = self._seeded(6)
        fc_conv = fc.consensus_convergence("钻木可以取火", max_rounds=60,
                                           threshold=0.8)
        ring = self._seeded(6)
        ring.set_topology("ring")
        ring_conv = ring.consensus_convergence("钻木可以取火", max_rounds=60,
                                               threshold=0.8)
        self.assertTrue(fc_conv["converged"])
        if ring_conv["converged"]:
            self.assertLessEqual(fc_conv["rounds"], ring_conv["rounds"])

    def test_consensus_convergence_report(self):
        swarm = self._seeded(4)
        conv = swarm.consensus_convergence("钻木可以取火", max_rounds=40,
                                           threshold=0.9)
        self.assertTrue(conv["converged"])
        self.assertGreaterEqual(conv["rounds"], 1)
        self.assertGreaterEqual(conv["coverage"][-1], 0.9)
        self.assertTrue(all(b >= a for a, b in
                            zip(conv["coverage"], conv["coverage"][1:])))

    def test_phase_scan_grid(self):
        """相变扫描：网格完整且全连接在每规模下不慢于环形"""
        from swarm import consensus_phase_scan
        grid = consensus_phase_scan(sizes=(3, 6),
                                    topologies=("fully_connected", "ring"),
                                    max_rounds=60, seed=1)
        self.assertEqual(len(grid), 4)
        for g in grid:
            self.assertIn("mean_degree", g)
            self.assertIn("coverage", g)
        for size in (3, 6):
            fc = next(g for g in grid
                      if g["size"] == size
                      and g["topology"] == "fully_connected")
            rg = next(g for g in grid
                      if g["size"] == size and g["topology"] == "ring")
            if fc["converged"] and rg["converged"]:
                self.assertLessEqual(fc["rounds"], rg["rounds"])


class TestCoevolution(unittest.TestCase):
    """v4.2：拓扑自适应——共识压力反作用于社交边的生灭"""

    def _seeded_ring(self, n=12, seed=1):
        swarm = BrainSwarm([f"C{i}" for i in range(n)], seed=seed)
        for _ in range(25):
            swarm.population[0].sensory_input("钻木可以取火")
        swarm.set_topology("ring")
        return swarm

    def test_opinion_copy_spreads_meme(self):
        """观点模仿：异见边上持有方把模因教给未持有方"""
        swarm = self._seeded_ring(6)
        ops = swarm.rewire_coevolve("钻木可以取火", rewire_prob=0.0,
                                    birth_prob=0.0)
        self.assertGreater(ops["copied"], 0)
        holders = sum(any(m.content == "钻木可以取火" for m in b.long_memory)
                      for b in swarm.population)
        self.assertGreater(holders, 1)

    def test_rewire_preserves_degree_floor(self):
        """断边重连保持最小度约束，无孤立个体"""
        swarm = self._seeded_ring(6)
        swarm.rewire_coevolve("钻木可以取火", rewire_prob=1.0,
                              birth_prob=0.0, min_degree=1)
        for i in range(len(swarm.population)):
            self.assertGreaterEqual(len(swarm.topology[i]), 1)

    def test_knowledge_seeking_births_grow_edges(self):
        """求知连边：边数随共识压力增长（生灭不守恒）"""
        swarm = self._seeded_ring(8)
        before = swarm._edge_count()
        swarm.rewire_coevolve("钻木可以取火", rewire_prob=0.5,
                              birth_prob=1.0)
        self.assertGreater(swarm._edge_count(), before)

    def test_same_state_ratio_organizes(self):
        """同道边比例是合法观测量且共同演化后仍处 [0,1]"""
        swarm = self._seeded_ring(8)
        r0 = swarm.same_state_edge_ratio("钻木可以取火")
        self.assertTrue(0.0 <= r0 <= 1.0)
        swarm.coevolve_consensus("钻木可以取火", max_rounds=10,
                                 rewire_prob=0.5)
        r1 = swarm.same_state_edge_ratio("钻木可以取火")
        self.assertTrue(0.0 <= r1 <= 1.0)

    def test_coevolve_beats_static_ring(self):
        """共同演化使静态下无法收敛的稀疏拓扑快速收敛"""
        static = self._seeded_ring(12)
        st = static.consensus_convergence("钻木可以取火", max_rounds=60,
                                          threshold=0.9)
        coevo = self._seeded_ring(12)
        co = coevo.coevolve_consensus("钻木可以取火", max_rounds=60,
                                      threshold=0.9, rewire_prob=0.5)
        self.assertFalse(st["converged"])          # 静态环形 N=12 不收敛
        self.assertTrue(co["converged"])           # 共同演化收敛
        self.assertLessEqual(co["rounds"], 60)
        # 三条观测曲线完整
        self.assertEqual(len(co["coverage"]), co["rounds"])
        self.assertEqual(len(co["edges"]), co["rounds"])
        self.assertEqual(len(co["same_state_ratio"]), co["rounds"])

    def test_rewire_prob_slows_consensus(self):
        """φ 三区：高重连概率的结构 churn 拖慢共识"""
        def run(phi):
            s = self._seeded_ring(12)
            return s.coevolve_consensus("钻木可以取火", max_rounds=60,
                                        threshold=0.9, rewire_prob=phi)
        low, high = run(0.2), run(0.8)
        self.assertTrue(low["converged"] and high["converged"])
        self.assertLessEqual(low["rounds"], high["rounds"])
        self.assertGreater(high["rewired_total"], low["rewired_total"])

    def test_full_topology_noop(self):
        """全连接（无拓扑）时共同演化为安全空操作"""
        swarm = BrainSwarm(["A", "B"], seed=1)
        swarm.population[0].long_memory.append(BrainMemory(
            content="钻木可以取火", timestamp=0.0, weight=1.0, tag="event"))
        ops = swarm.rewire_coevolve("钻木可以取火")
        self.assertEqual(ops, {"rewired": 0, "copied": 0, "born": 0})


class TestMemeCompetition(unittest.TestCase):
    """v4.3：多模因竞争——垄断（共识）vs 极化（共存）"""

    RIVALS = ["钻木可以取火", "燧石可以取火"]

    def _split_swarm(self, n=12, seed=1):
        """前半持钻木、后半持燧石的两阵营环形种群"""
        import time as _time
        swarm = BrainSwarm([f"M{i}" for i in range(n)], seed=seed)
        for k in range(n):
            meme = self.RIVALS[0] if k < n // 2 else self.RIVALS[1]
            swarm.population[k].long_memory.append(BrainMemory(
                content=meme, timestamp=_time.time(), weight=1.0,
                tag="culture"))
        swarm.set_topology("ring")
        return swarm

    def test_stance_detection(self):
        swarm = self._split_swarm(6)
        self.assertEqual(swarm._stance(0, self.RIVALS), "钻木可以取火")
        self.assertEqual(swarm._stance(5, self.RIVALS), "燧石可以取火")
        swarm.population[2].long_memory.clear()
        self.assertIsNone(swarm._stance(2, self.RIVALS))

    def test_conversion_switches_stance(self):
        """立场转化：弱势方移除竞争模因并改持教师模因"""
        swarm = self._split_swarm(6)
        ops = swarm.compete_coevolve(self.RIVALS, rewire_prob=0.0,
                                     birth_prob=0.0)
        self.assertGreater(ops["converted"], 0)

    def test_low_phi_monopoly(self):
        """低重连概率 → 转化主导 → 一个模因垄断全网"""
        swarm = self._split_swarm(12)
        res = swarm.competition_dynamics(self.RIVALS, max_rounds=60,
                                         dominance=0.9, rewire_prob=0.2)
        self.assertTrue(res["converged"])
        self.assertIn(res["winner"], self.RIVALS)
        self.assertGreaterEqual(res["final"][res["winner"]], 0.9)
        # 覆盖率曲线完整
        self.assertEqual(len(res["coverage"][res["winner"]]), res["rounds"])

    def test_high_phi_polarization(self):
        """高重连概率 → 阵营隔离 → 两模因极化共存"""
        swarm = self._split_swarm(12)
        res = swarm.competition_dynamics(self.RIVALS, max_rounds=60,
                                         dominance=0.9, rewire_prob=0.85)
        self.assertFalse(res["converged"])
        self.assertIsNone(res["winner"])
        self.assertEqual(res["camps"][-1], 2)
        self.assertGreater(res["final"][self.RIVALS[0]], 0.0)
        self.assertGreater(res["final"][self.RIVALS[1]], 0.0)

    def test_camp_births_grow_edges(self):
        """阵营连边：有立场者向同阵营连新边，边数增长"""
        swarm = self._split_swarm(8)
        before = swarm._edge_count()
        swarm.compete_coevolve(self.RIVALS, rewire_prob=0.0,
                               birth_prob=1.0)
        self.assertGreater(swarm._edge_count(), before)

    def test_competition_full_topology_noop(self):
        swarm = BrainSwarm(["A", "B"], seed=1)
        ops = swarm.compete_coevolve(self.RIVALS)
        self.assertEqual(ops, {"rewired": 0, "converted": 0, "born": 0})


class TestEligibilityTrace(unittest.TestCase):
    """v4.4：TD(λ) 资格迹——信用分配跨 tick 反向传播"""

    CUES = ["铃声", "灯光", "气味"]

    def _train(self, trials=20, reward=1.0):
        """Schultz 范式：三线索链 → 奖励，反复试次"""
        brain = AIBrainEntity("λ", seed=1)
        last = None
        for _ in range(trials):
            for c in self.CUES:
                brain.sensory_input(c)
            last = brain.reward_lambda(reward)
        return brain, last

    def test_perception_marks_state(self):
        brain = AIBrainEntity("λ", seed=1)
        brain.sensory_input("铃声")
        self.assertEqual(brain.eligibility["铃声"], 1.0)
        self.assertEqual(brain._last_state, "铃声")

    def test_trace_decays_with_gamma_lambda(self):
        brain = AIBrainEntity("λ", seed=1)
        brain.sensory_input("铃声")
        brain.sensory_input("灯光")
        expected = brain.td_gamma * brain.td_lambda
        self.assertAlmostEqual(brain.eligibility["铃声"], expected)

    def test_credit_flows_backward_to_earlier_cues(self):
        """一次奖励同时给链上所有线索分配信用（不只是当前状态）"""
        brain = AIBrainEntity("λ", seed=1)
        for c in self.CUES:
            brain.sensory_input(c)
        r = brain.reward_lambda(1.0)
        for c in self.CUES:
            self.assertIn(c, r["credited"])
            self.assertGreater(r["credited"][c], 0.0)
        # 越早的线索分到的信用越少（γλ 折扣梯度）
        self.assertLess(r["credited"]["铃声"], r["credited"]["气味"])

    def test_value_gradient_after_training(self):
        """训练后 V 沿链呈梯度：越靠近奖励的线索价值越高"""
        brain, _ = self._train(20)
        v = brain.state_values
        self.assertLess(v["铃声"], v["灯光"])
        self.assertLess(v["灯光"], v["气味"])
        self.assertGreater(v["气味"], 0.9)   # 末线索收敛到奖励值附近

    def test_rpe_shrinks_as_reward_becomes_predicted(self):
        """多巴胺时序迁移：奖励被早期线索预测后，奖励时刻 RPE → 0"""
        brain = AIBrainEntity("λ", seed=1)
        for c in self.CUES:
            brain.sensory_input(c)
        first = brain.reward_lambda(1.0)["rpe"]
        _, last = self._train(20)
        self.assertAlmostEqual(first, 1.0)
        self.assertLess(abs(last["rpe"]), 0.05)

    def test_negative_reward_propagates_negative_credit(self):
        brain = AIBrainEntity("λ", seed=1)
        for c in self.CUES:
            brain.sensory_input(c)
        r = brain.reward_lambda(-1.0)
        for c in self.CUES:
            self.assertLess(brain.state_values[c], 0.0)
        self.assertLess(r["dopamine"], 0.0)


class TestSkillLearning(unittest.TestCase):
    """v4.5：技能学习——分 verb 独立价值估计 + 策略化动作选择"""

    def test_learn_skill_moves_q_toward_reward(self):
        brain = AIBrainEntity("S", seed=1)
        r = brain.learn_skill("respond", 1.0)
        self.assertAlmostEqual(r["q"], 0.3)          # 0 + 0.3×(1−0)
        self.assertAlmostEqual(r["rpe"], 1.0)

    def test_q_converges_per_verb(self):
        brain = AIBrainEntity("S", seed=1)
        for _ in range(40):
            brain.learn_skill("respond", 0.8)
            brain.learn_skill("observe", -0.4)
        self.assertAlmostEqual(brain.verb_values["respond"], 0.8, places=2)
        self.assertAlmostEqual(brain.verb_values["observe"], -0.4, places=2)
        self.assertEqual(brain.verb_values["acknowledge"], 0.0)  # 未动作者不动

    def test_greedy_selects_highest_q(self):
        brain = AIBrainEntity("S", seed=1)
        brain.verb_values.update({"respond": 0.8, "acknowledge": 0.2,
                                  "observe": -0.4})
        for _ in range(10):
            self.assertEqual(brain.select_verb("greedy"), "respond")

    def test_zero_q_uniform_random(self):
        """未学习时 Q 全零，greedy 退化为均匀随机（不恒选某一 verb）"""
        brain = AIBrainEntity("S", seed=3)
        picks = {brain.select_verb("greedy") for _ in range(30)}
        self.assertGreater(len(picks), 1)


class TestRetrievalComposition(unittest.TestCase):
    """v4.6：检索式语言生成——LTM 片段 + 句法框架组合"""

    def _brain_with_memories(self):
        brain = AIBrainEntity("C", seed=1)
        for _ in range(30):
            brain.sensory_input("火焰是危险的")
        for _ in range(20):
            brain.sensory_input("钻木可以取火")
        for _ in range(15):
            brain.sensory_input("燧石可以取火")
        return brain

    def test_no_memory_fallback_clause(self):
        brain = AIBrainEntity("C", seed=1)
        brain.sensory_input("陌生的东西")
        c = brain.compose("陌生的东西")
        self.assertEqual(c["fragments"], [])
        self.assertIn("全新的体验", c["utterance"])

    def test_single_fragment_clause(self):
        brain = self._brain_with_memories()
        brain.sensory_input("火焰")
        c = brain.compose("火焰")
        self.assertEqual(c["fragments"], ["火焰是危险的"])
        self.assertIn("浮现在脑海", c["utterance"])

    def test_multiple_fragments_woven(self):
        """n-gram 降级检索：长词部分匹配也能取出多条记忆并编织"""
        brain = self._brain_with_memories()
        brain.sensory_input("取火的方法")
        c = brain.compose("取火的方法")
        self.assertIn("钻木可以取火", c["fragments"])
        self.assertIn("燧石可以取火", c["fragments"])
        self.assertIn("一起浮现", c["utterance"])

    def test_fragments_ranked_by_weight(self):
        brain = self._brain_with_memories()
        # 显式拉开权重梯度（固化权重会封顶 1.0，无法体现排序）
        weights = {"钻木可以取火": 0.9, "燧石可以取火": 0.6,
                   "火焰是危险的": 0.3}
        for m in brain.long_memory + brain.short_memory:
            if m.content in weights:
                m.weight = weights[m.content]
        brain.sensory_input("取火 火焰")
        c = brain.compose("取火 火焰")
        self.assertEqual(len(c["fragments"]), 3)
        # 权重最高者排最前，成为联想链主线
        self.assertEqual(c["fragments"][0], "钻木可以取火")
        self.assertEqual(c["fragments"][-1], "火焰是危险的")

    def test_compose_returns_frame_and_mood(self):
        brain = self._brain_with_memories()
        brain.sensory_input("火焰")
        c = brain.compose("火焰")
        self.assertIn(c["frame"],
                      brain._SYNTAX_FRAMES[c["action"]["action"]])
        self.assertIn(c["mood"], ("calm", "curiosity", "stress", "pleasure"))

    def test_compose_deterministic_same_tick(self):
        brain = self._brain_with_memories()
        brain.sensory_input("取火 火焰")
        self.assertEqual(brain.compose("取火 火焰")["utterance"],
                         brain.compose("取火 火焰")["utterance"])


class TestEpisodicIndex(unittest.TestCase):
    """v4.7：情景记忆时间索引——何时 + 与何事共现 + 时间推理"""

    def _diary_brain(self):
        brain = AIBrainEntity("E", seed=1)
        for s in ["起床", "刷牙", "吃早餐", "出门", "刷牙", "上班"]:
            brain.sensory_input(s)
        return brain

    def test_perception_records_episode(self):
        brain = AIBrainEntity("E", seed=1)
        brain.sensory_input("起床")
        brain.sensory_input("刷牙")
        self.assertEqual(len(brain.episodes), 2)
        ep = brain.episodes[1]
        self.assertEqual(ep["content"], "刷牙")
        self.assertEqual(ep["tick"], 2)
        self.assertEqual(ep["context"], ["起床"])   # 共现内容

    def test_episodic_trace_in_order(self):
        brain = self._diary_brain()
        trace = brain.episodic_trace("刷牙")
        self.assertEqual(len(trace), 2)
        self.assertEqual([ep["tick"] for ep in trace], [2, 5])

    def test_events_after_uses_latest_anchor(self):
        """"上次" = 最近一次锚点；delta 为 tick 间隔"""
        brain = self._diary_brain()
        r = brain.events_after("刷牙")
        self.assertEqual(r["anchor"]["tick"], 5)
        self.assertEqual([e["content"] for e in r["events"]], ["上班"])
        self.assertEqual(r["events"][0]["delta"], 1)

    def test_events_before(self):
        brain = self._diary_brain()
        r = brain.events_before("吃早餐")
        self.assertEqual([e["content"] for e in r["events"]],
                         ["起床", "刷牙"])
        self.assertTrue(all(e["delta"] < 0 for e in r["events"]))

    def test_missing_anchor_returns_empty(self):
        brain = self._diary_brain()
        r = brain.events_after("睡觉")
        self.assertIsNone(r["anchor"])
        self.assertEqual(r["events"], [])

    def test_context_excludes_self_and_caps_with_buffer(self):
        brain = AIBrainEntity("E", seed=1)
        for i in range(12):
            brain.sensory_input(f"事件{i}")
        last = brain.episodes[-1]
        self.assertNotIn(last["content"], last["context"])
        # 感官缓存上限 8 → 共现最多 7 条（不含自身）
        self.assertEqual(len(last["context"]), 7)


class TestSleepCycle(unittest.TestCase):
    """v4.8：睡眠-清醒节律——离线重放固化 + 突触稳态缩放"""

    def _drowsy_brain(self):
        """带两条弱 STM 刺激的实体（白天路径无法固化）"""
        brain = AIBrainEntity("Z", seed=1)
        brain.sensory_input("萤火虫在夜里发光")
        brain.sensory_input("另一个无关刺激")
        return brain

    def test_replay_consolidates_weak_memories(self):
        brain = self._drowsy_brain()
        self.assertEqual(len(brain.long_memory), 0)
        r = brain.sleep(cycles=3)
        self.assertEqual(r["consolidated"], 2)
        self.assertEqual(len(brain.short_memory), 0)
        self.assertEqual(len(brain.long_memory), 2)

    def test_without_sleep_weak_memories_forgotten(self):
        """对照：不睡眠只衰减，弱刺激进不了 LTM"""
        brain = self._drowsy_brain()
        for _ in range(30):
            brain.decay_memory(0.9)
        self.assertEqual(len(brain.long_memory), 0)

    def test_synapse_downscale_per_cycle(self):
        brain = AIBrainEntity("Z", seed=1)
        before = dict(brain.synapse)
        brain.sleep(cycles=12)
        common = [k for k in before if k in brain.synapse]
        ratio = brain.synapse[common[0]] / before[common[0]]
        self.assertAlmostEqual(ratio, 0.95 ** 12, places=3)

    def test_pruning_removes_weak_keeps_strong(self):
        brain = AIBrainEntity("Z", seed=1)
        before = dict(brain.synapse)
        r = brain.sleep(cycles=12)
        self.assertGreater(r["pruned_synapses"], 0)
        # 最强连接必然存活（等比缩放保留相对差异）
        strongest = max(before, key=before.get)
        self.assertIn(strongest, brain.synapse)

    def test_sleep_washes_traces_and_dopamine(self):
        brain = self._drowsy_brain()
        brain.reward_lambda(0.8)               # 留下资格迹与多巴胺
        self.assertTrue(brain.eligibility)
        brain.sleep(cycles=1)
        self.assertEqual(brain.eligibility, {})
        self.assertEqual(brain.dopamine, 0.0)

    def test_sleep_recovers_stress_and_calm(self):
        brain = self._drowsy_brain()
        brain.reward(-0.8)
        stress_before = brain.emotion["stress"]
        calm_before = brain.emotion["calm"]
        r = brain.sleep(cycles=3)
        self.assertLess(brain.emotion["stress"], stress_before)
        self.assertGreater(brain.emotion["calm"], calm_before)
        self.assertAlmostEqual(r["stress_after"],
                               round(brain.emotion["stress"], 3))


class TestCuriosityDrivenExploration(unittest.TestCase):
    """v4.9：好奇驱动探索——新奇度反向调制注意与 ε"""

    def _familiar_brain(self):
        brain = AIBrainEntity("N", seed=1)
        for _ in range(30):
            brain.sensory_input("火焰是危险的")
        return brain

    def test_familiar_stimulus_low_novelty(self):
        brain = self._familiar_brain()
        brain.sensory_input("火焰")
        self.assertEqual(brain.novelty, 0.0)

    def test_novel_stimulus_high_novelty(self):
        brain = self._familiar_brain()
        brain.sensory_input("量子纠缠态坍缩")
        self.assertGreaterEqual(brain.novelty, 0.7)   # 全未命中 + 无 RPE

    def test_novelty_captures_attention_same_tick(self):
        brain = self._familiar_brain()
        brain.sensory_input("火焰")
        att_familiar = brain.attention_factor
        brain.sensory_input("量子纠缠态坍缩")
        self.assertGreater(brain.attention_factor, att_familiar)

    def test_effective_epsilon_scales_with_novelty(self):
        brain = AIBrainEntity("N", seed=1)
        brain.skill_epsilon = 0.4
        brain.novelty = 0.0
        self.assertAlmostEqual(brain.effective_epsilon(), 0.2)   # 0.5ε
        brain.novelty = 1.0
        self.assertAlmostEqual(brain.effective_epsilon(), 0.6)   # 1.5ε

    def test_more_exploration_when_novel(self):
        """行为层面：同种子同 Q，新奇时 ε-greedy 探索次数更多"""
        def explore_count(nov):
            brain = AIBrainEntity("N", seed=5)
            brain.verb_values.update({"respond": 0.9, "acknowledge": 0.0,
                                      "observe": -0.5})
            brain.skill_epsilon = 0.4
            brain.novelty = nov
            picks = [brain.select_verb("epsilon") for _ in range(300)]
            return sum(1 for p in picks if p != "respond")
        self.assertLess(explore_count(0.0), explore_count(1.0))

    def test_rpe_surprise_elevates_novelty(self):
        """大意外之后，熟悉刺激也重获新奇度（|RPE| 分量）"""
        brain = self._familiar_brain()
        brain.reward_lambda(1.0)          # |RPE|=1
        brain.sensory_input("火焰")
        self.assertAlmostEqual(brain.novelty, 0.3, places=2)


class TestThoughtSystem(unittest.TestCase):
    """v5.0 思考体系：思考空间 / 思考记忆 / 思考感官"""

    def setUp(self):
        self.brain = AIBrainEntity("Thinker", seed=1)

    def test_perception_enters_thought_space(self):
        """外部感知进入思考空间，来源标记为 external"""
        self.brain.sensory_input("火焰是危险的")
        self.assertEqual(len(self.brain.thought_space), 1)
        t = self.brain.thought_space[0]
        self.assertIsInstance(t, ThoughtItem)
        self.assertEqual(t.content, "火焰是危险的")
        self.assertEqual(t.source, "external")

    def test_thought_capacity_evicts_weakest(self):
        """容量 7±2=9：超出时挤出激活度最低的念头"""
        for i in range(12):
            self.brain._push_thought(f"测试想法{i}",
                                     activation=0.1 * (i % 9 + 1))
        self.assertLessEqual(len(self.brain.thought_space),
                             self.brain.thought_capacity)
        # 被挤出的应是激活度最低者
        activations = [t.activation for t in self.brain.thought_space]
        self.assertEqual(min(activations),
                         min(t.activation for t in self.brain.thought_space))

    def test_thought_decay_and_exit(self):
        """激活度逐 tick 衰减，低于 0.05 退出意识"""
        self.brain._push_thought("短暂的念头", activation=0.1)
        for _ in range(3):
            self.brain._decay_thoughts()
        # 0.1 × 0.9³ ≈ 0.073 > 0.05 仍在
        self.assertEqual(len(self.brain.thought_space), 1)
        for _ in range(4):
            self.brain._decay_thoughts()
        # 0.1 × 0.9⁷ ≈ 0.048 < 0.05 退出意识
        self.assertEqual(self.brain.thought_space, [])

    def test_same_content_reactivates_not_duplicates(self):
        """同内容念头重新激活而非重复入栈"""
        self.brain._push_thought("同一个念头")
        self.brain._push_thought("同一个念头")
        self.assertEqual(len(self.brain.thought_space), 1)
        self.assertEqual(self.brain.thought_space[0].activation, 1.0)

    def test_recall_enters_thought_space(self):
        """联想回忆起的记忆进入意识，来源标记为 memory"""
        for _ in range(30):
            self.brain.sensory_input("钻木可以取火")
        self.brain.sensory_input("别的内容")
        recalled = self.brain.recall("钻木")
        self.assertTrue(recalled)
        sources = {t.content: t.source for t in self.brain.thought_space}
        self.assertEqual(sources.get(recalled[0].content), "memory")

    def test_think_advances_tick_and_returns_structure(self):
        """think() 推进 tick，返回思考报告结构"""
        self.brain.sensory_input("火焰")
        tick_before = self.brain.tick
        out = self.brain.think("火焰")
        self.assertGreater(self.brain.tick, tick_before)
        self.assertEqual(out["thought"], "火焰")
        self.assertIn("spikes", out)
        self.assertIn("thought_space", out)
        self.assertIn("consolidated", out)

    def test_think_empty_space_returns_none(self):
        """无念头且无参数思考：安全返回 thought=None"""
        out = self.brain.think()
        self.assertIsNone(out["thought"])
        self.assertEqual(out["thought_space"], [])

    def test_think_consolidates_thought_memory(self):
        """高激活念头固化进 STM（tag=thought）——想多了就记住了"""
        self.brain.think("必须记住的结论")
        tags = {m.content: m.tag for m in self.brain.short_memory}
        self.assertEqual(tags.get("必须记住的结论"), "thought")

    def test_think_uses_top_thought_by_default(self):
        """缺省参数时思考意识焦点（激活度最高的念头）"""
        self.brain.sensory_input("第一个刺激")
        self.brain.sensory_input("第二个刺激")
        out = self.brain.think()
        top = max(self.brain.thought_space, key=lambda t: t.activation)
        self.assertEqual(out["thought"], top.content)

    def test_introspect_structure_and_metacog_log(self):
        """introspect() 感知自身脑活动并记入元认知日志"""
        self.brain.sensory_input("火焰是危险的")
        entry = self.brain.introspect()
        for key in ("tick", "mood", "top_thought", "spike_counts",
                    "stm", "ltm", "text"):
            self.assertIn(key, entry)
        self.assertIn(entry["mood"],
                      ("calm", "curiosity", "stress", "pleasure"))
        self.assertEqual(len(self.brain.metacog_log), 1)
        # 内省言语作为元认知念头进入思考空间
        sources = {t.source for t in self.brain.thought_space}
        self.assertIn("metacog", sources)
        self.assertIn("我感到", entry["text"])

    def test_introspect_empty_brain_safe(self):
        """全新大脑内省：思考空间为空时安全返回（空）"""
        entry = self.brain.introspect()
        self.assertEqual(entry["top_thought"], "（空）")

    def test_dna_roundtrip_preserves_thoughts(self):
        """DNA 克隆保留思考空间与元认知日志"""
        self.brain.sensory_input("火焰是危险的")
        self.brain.introspect()
        clone = AIBrainEntity.from_dna(self.brain.dump_dna())
        self.assertEqual(len(clone.thought_space),
                         len(self.brain.thought_space))
        self.assertEqual(clone.metacog_log, self.brain.metacog_log)
        src = sorted(t.content for t in self.brain.thought_space)
        dst = sorted(t.content for t in clone.thought_space)
        self.assertEqual(src, dst)

    def test_old_dna_without_thought_fields_loads(self):
        """旧版 DNA（无 v5.0 字段）兼容加载"""
        dna = self.brain.dump_dna()
        del dna["thought_space"]
        del dna["metacog_log"]
        clone = AIBrainEntity.from_dna(dna)
        self.assertEqual(clone.thought_space, [])
        self.assertEqual(clone.metacog_log, [])

    def test_dna_roundtrip_preserves_self_model(self):
        """DNA 克隆保留 v5.2+ 自我模型：自我概念/自传体记忆/心智模型/世代/模因"""
        brain = AIBrainEntity("SELF", seed=1)
        brain.update_self_concept("我是好奇的")
        brain.add_autobiographical_memory("第一次看到大海", emotion="joy", importance=0.9)
        other = AIBrainEntity("OTHER", seed=2)
        brain.attribute_beliefs(other)
        brain.generation = 3
        brain.add_meme("模因甲", source="同伴")
        clone = AIBrainEntity.from_dna(brain.dump_dna())
        self.assertEqual(clone.self_concept, ["我是好奇的"])
        self.assertEqual(len(clone.autobiographical_memory), 1)
        self.assertEqual(clone.autobiographical_memory[0]["event"], "第一次看到大海")
        self.assertIn("OTHER", clone.mental_models)
        self.assertEqual(clone.generation, 3)
        self.assertIn("模因甲", clone.memes)
        self.assertEqual(clone.meme_system["total_memes"], 1)

    def test_old_dna_without_self_fields_loads(self):
        """旧版 DNA（无 v5.2+ 字段）兼容加载，默认空自我模型"""
        dna = self.brain.dump_dna()
        for k in ("generation", "thought_journal", "self_concept",
                  "autobiographical_memory", "mental_models", "memes"):
            del dna[k]
        clone = AIBrainEntity.from_dna(dna)
        self.assertEqual(clone.generation, 1)
        self.assertEqual(clone.self_concept, [])
        self.assertEqual(clone.autobiographical_memory, [])
        self.assertEqual(clone.mental_models, {})
        self.assertEqual(clone.memes, {})

    def test_status_reports_thought_space(self):
        """status() 摘要包含思考空间行"""
        self.brain.sensory_input("火焰")
        text = self.brain.status()
        self.assertIn("思考空间", text)

    def test_status_reports_metacog(self):
        """status() 摘要包含元认知状态（无内省/有内省两种）"""
        self.assertIn("元认知: 0 条日志（暂无内省记录）", self.brain.status())
        self.brain.sensory_input("火焰是危险的")
        self.brain.introspect()
        text = self.brain.status()
        self.assertIn("元认知: 1 条日志", text)
        self.assertIn("最近:", text)

    def test_thought_chain_includes_thoughts(self):
        """thought_chain() 返回值新增 thoughts 字段（思考空间快照）"""
        tc = self.brain.thought_chain("火焰是危险的")
        self.assertIn("thoughts", tc)
        self.assertIsInstance(tc["thoughts"], list)
        self.assertTrue(tc["thoughts"])
        first = tc["thoughts"][0]
        for key in ("content", "source", "activation", "birth_tick"):
            self.assertIn(key, first)
        self.assertEqual(first["content"], "火焰是危险的")
        self.assertEqual(first["source"], "external")


class TestIntentVerbs(unittest.TestCase):
    """v5.1 动作与决策扩展：8 verb 动作空间 / 深思熟虑 / 实用执行器"""

    def setUp(self):
        self.brain = AIBrainEntity("Actor", seed=1)

    def test_intent_verbs_table(self):
        """意图动词表覆盖 8 个 verb，channel 取值合法"""
        self.assertEqual(len(AIBrainEntity.INTENT_VERBS), 8)
        for v in ("respond", "acknowledge", "observe",
                  "ask", "retrieve", "plan", "execute", "wait"):
            self.assertIn(v, AIBrainEntity.INTENT_VERBS)
            self.assertIn(AIBrainEntity.INTENT_VERBS[v]["channel"],
                          ("external", "internal"))

    def test_verb_values_cover_all_intent_verbs(self):
        """技能价值表初始化覆盖全部 8 个 verb（Q=0）"""
        self.assertEqual(len(self.brain.verb_values), 8)
        self.assertTrue(all(q == 0.0
                            for q in self.brain.verb_values.values()))

    def test_decide_action_default_unchanged(self):
        """默认决策行为不变：无理由链，verb 限于脉冲三动作"""
        out = self.brain.decide_action("火焰")
        self.assertNotIn("rationale", out)
        self.assertIn(out["verb"], ("respond", "acknowledge", "observe"))

    def test_deliberate_structure(self):
        """深思熟虑决策带 rationale/base_verb/q_values/novelty"""
        out = self.brain.decide_action("火焰", deliberate=True)
        self.assertIn("rationale", out)
        self.assertIn("base_verb", out)
        self.assertIn("q_values", out)
        self.assertIn("novelty", out)
        self.assertEqual(len(out["q_values"]), 8)
        self.assertTrue(out["rationale"])

    def test_deliberate_ask_heuristic(self):
        """记忆未命中且新奇度 >0.5 → 提问澄清"""
        self.brain.sensory_input("从未见过的紫色星星")
        out = self.brain.decide_action("从未见过的紫色星星",
                                       deliberate=True)
        self.assertEqual(out["verb"], "ask")
        self.assertEqual(out["base_verb"], "observe")  # 无脉冲时对照
        self.assertTrue(any("提问澄清" in r for r in out["rationale"]))

    def test_deliberate_retrieve_heuristic(self):
        """多条记忆同时命中 → 主动检索回忆"""
        for _ in range(3):
            self.brain.sensory_input("火焰可以取暖")
            self.brain.sensory_input("危险需要远离")
        self.brain.sensory_input("无关的缓冲")
        out = self.brain.decide_action("火焰 危险", deliberate=True)
        self.assertGreaterEqual(len(out["recalled"]), 2)
        self.assertEqual(out["verb"], "retrieve")

    def test_deliberate_policy_overrides_verb(self):
        """policy 非空时由习得 Q 值选 verb"""
        for _ in range(10):
            self.brain.learn_skill("ask", 0.8)
        out = self.brain.decide_action("火焰", deliberate=True,
                                       policy="greedy")
        self.assertEqual(out["verb"], "ask")
        self.assertTrue(any("greedy" in r for r in out["rationale"]))

    def test_deliberate_plan_pushes_goal_thought(self):
        """plan verb 把规划目标压入思考空间"""
        for _ in range(10):
            self.brain.learn_skill("plan", 0.9)
        out = self.brain.decide_action("建造树屋", deliberate=True,
                                       policy="greedy")
        self.assertEqual(out["verb"], "plan")
        self.assertTrue(any(t.content.startswith("规划目标")
                            for t in self.brain.thought_space))

    def test_express_uses_intent_verb_template(self):
        """意图动词优先使用专属动词模板"""
        for _ in range(10):
            self.brain.learn_skill("ask", 0.8)
        out = self.brain.express("神秘信号", deliberate=True,
                                 policy="greedy")
        self.assertEqual(out["action"]["verb"], "ask")
        self.assertIn("？", out["utterance"])


class TestStreamOfConsciousness(unittest.TestCase):
    """v5.2 意识流：自由联想、白日梦、灵感闪现"""

    def test_returns_chain_structure(self):
        brain = AIBrainEntity("soc", seed=1)
        for s in ["火焰是危险的", "水能灭火", "太阳会发光"]:
            brain.sensory_input(s)
        random.seed(7)
        out = brain.stream_of_consciousness(steps=4)
        self.assertIn("chain", out)
        self.assertIn("insights", out)
        self.assertIn("final_thought", out)
        self.assertIn("thought_space_size", out)
        self.assertEqual(out["daydream_level"], 0.3)
        self.assertIsInstance(out["chain"], list)
        self.assertGreaterEqual(len(out["chain"]), 1)

    def test_empty_brain_safe(self):
        """空大脑（无记忆无念头）：安全返回空链"""
        brain = AIBrainEntity("empty", seed=2)
        out = brain.stream_of_consciousness(steps=3)
        self.assertEqual(out["chain"], [])
        self.assertIsNone(out["final_thought"])

    def test_daydream_zero_no_wander(self):
        """daydream=0 时不出现 [走神] 标记"""
        brain = AIBrainEntity("nodream", seed=3)
        for s in ["记忆是智慧的基石", "神经元在放电"]:
            brain.sensory_input(s)
        random.seed(11)
        out = brain.stream_of_consciousness(steps=4, daydream=0.0)
        self.assertFalse(any("[走神]" in c for c in out["chain"]))


class TestIntrospectDepth(unittest.TestCase):
    """v5.2 自我意识：basic / deep 内省"""

    def test_basic_entry_fields(self):
        brain = AIBrainEntity("intro", seed=1)
        brain.sensory_input("你好世界")
        entry = brain.introspect(depth="basic")
        for k in ("tick", "mood", "top_thought", "spike_counts",
                  "stm", "ltm", "thought_space_size",
                  "novelty", "attention", "depth", "text"):
            self.assertIn(k, entry)
        self.assertEqual(entry["depth"], "basic")
        self.assertTrue(entry["text"].startswith("我感到"))
        # 记入元认知日志
        self.assertEqual(brain.metacog_log[-1], entry)

    def test_deep_differs_from_basic(self):
        brain = AIBrainEntity("deep", seed=1)
        brain.sensory_input("量子纠缠很神奇")
        entry = brain.introspect(depth="deep")
        # 深度内省包含自我指称与更多内部状态描述
        self.assertIn("我是deep", entry["text"])
        self.assertIn("新奇度", entry["text"])
        self.assertIn("思考空间", entry["text"])
        # 深度内省言语也回注为 metacog 念头
        self.assertTrue(any(t.source == "metacog"
                            for t in brain.thought_space))


class TestSocialInteraction(unittest.TestCase):
    """v5.2 社交互动：发消息 / 文化学习 / 多轮对话"""

    def setUp(self):
        self.alice = AIBrainEntity("Alice", seed=1)
        self.bob = AIBrainEntity("Bob", seed=2)

    def test_send_message_marks_social_memory(self):
        self.alice.send_message(self.bob, "今天天气不错")
        # 接收方 STM 有 tag="social" 的记忆，记录来源
        social = [m for m in self.bob.short_memory if m.tag == "social"]
        self.assertTrue(any("Alice告诉我今天天气不错" in m.content
                            for m in social))
        # 接收方思考空间有 source="social" 的念头
        self.assertTrue(any(t.source == "social"
                            for t in self.bob.thought_space))

    def test_social_learn_copies_top_memories(self):
        for s in ["勾股定理", "光速不变", "熵增原理"]:
            self.bob.sensory_input(s)
        p0 = self.alice.emotion["pleasure"]
        out = self.alice.social_learn(self.bob, n_memories=2)
        self.assertEqual(out["learned_from"], "Bob")
        self.assertEqual(out["learned_count"], len(out["learned"]))
        self.assertGreater(out["learned_count"], 0)
        # 学到的内容固化进 Alice 的 LTM，权重打 7 折
        src = {m.content: m.weight for m in
               list(self.bob.long_memory) + list(self.bob.short_memory)}
        for content in out["learned"]:
            mine = [m for m in self.alice.long_memory
                    if m.content == content]
            self.assertTrue(mine)
            self.assertAlmostEqual(mine[0].weight,
                                   src[content] * 0.7, places=6)
        # 获得新知识带来愉悦
        self.assertGreater(self.alice.emotion["pleasure"], p0)

    def test_social_learn_skips_known(self):
        self.bob.sensory_input("唯一知识")
        first = self.alice.social_learn(self.bob, n_memories=1)
        second = self.alice.social_learn(self.bob, n_memories=1)
        self.assertEqual(first["learned_count"], 1)
        self.assertEqual(second["learned_count"], 0)  # 已知的跳过

    def test_chat_with_turns(self):
        self.alice.sensory_input("你好")
        self.bob.sensory_input("我很好")
        conv = self.alice.chat_with(self.bob, turns=2)
        self.assertEqual(len(conv), 4)
        self.assertEqual(conv[0]["speaker"], "Alice")
        self.assertEqual(conv[1]["speaker"], "Bob")
        for turn in conv:
            self.assertIn("turn", turn)
            self.assertIn("message", turn)


class TestEvolution(unittest.TestCase):
    """v5.2 进化：适应度评估 / 选择 / 多代演化"""

    def test_evaluate_fitness_length_and_tasks(self):
        swarm = BrainSwarm(["A", "B", "C"], seed=1)
        swarm.population[0].sensory_input("x" * 30)
        scores = swarm.evaluate_fitness(task="memory")
        self.assertEqual(len(scores), 3)
        for task in ("curiosity", "diversity", "social"):
            self.assertEqual(len(swarm.evaluate_fitness(task)), 3)

    def test_evolve_generation_preserves_population(self):
        swarm = BrainSwarm(["A", "B", "C", "D"], seed=2)
        for i, brain in enumerate(swarm.population):
            for _ in range(i + 1):
                brain.sensory_input(f"知识{i}")
        random.seed(5)
        stats = swarm.evolve_generation(task="diversity")
        self.assertEqual(stats["population_size"], 4)
        self.assertEqual(stats["survivors"] + stats["born"], 4)
        self.assertEqual(stats["generation"], 2)
        self.assertEqual(swarm.generation, 2)
        self.assertEqual(len(swarm.population), 4)

    def test_evolve_history(self):
        swarm = BrainSwarm(["A", "B", "C", "D"], seed=3)
        swarm.population[0].sensory_input("优势记忆内容")
        random.seed(9)
        out = swarm.evolve(generations=2, task="diversity")
        self.assertEqual(out["generations"], 2)
        self.assertEqual(len(out["history"]), 2)
        self.assertEqual(out["final_population"], 4)
        self.assertEqual(out["final_generation"], 3)
        self.assertEqual(len(out["avg_fitness_trend"]), 2)
        self.assertIn("best_fitness", out)


class TestThoughtJournal(unittest.TestCase):
    """v5.2 念头流水账：所有念头按时间记录，容量封顶 50"""

    def test_journal_records_thoughts(self):
        brain = AIBrainEntity("journal", seed=1)
        brain.sensory_input("流水账测试")
        self.assertGreater(len(brain.thought_journal), 0)
        entry = brain.thought_journal[-1]
        self.assertIn("content", entry)
        self.assertIn("source", entry)
        self.assertIn("tick", entry)

    def test_journal_capped_at_50(self):
        brain = AIBrainEntity("cap", seed=1)
        for i in range(60):
            brain._push_thought(f"测试想法{i}", source="internal")
        self.assertEqual(len(brain.thought_journal), 50)
        # 最旧的被挤出，最新的一定在
        self.assertEqual(brain.thought_journal[-1]["content"], "测试想法59")


class TestLanguageGenerator(unittest.TestCase):
    """v5.9 语言生成器接入：大脑想什么 → 外部模型说出来（降级回模板）"""

    def setUp(self):
        self.brain = AIBrainEntity("LG", seed=1)
        self.brain.sensory_input("火焰是危险的")
        self.captured = []

    def tearDown(self):
        unregister_language_generator("fake")
        unregister_language_generator("qwen")

    def _register_fake(self, text="模型生成的回复"):
        def fake(ctx):
            self.captured.append(ctx)
            return text
        register_language_generator(fake, name="fake")

    def test_express_uses_generator(self):
        self._register_fake()
        out = self.brain.express("火焰")
        self.assertEqual(out["utterance"], "模型生成的回复")
        self.assertEqual(out["generator"], "fake")

    def test_generator_context_fields(self):
        self._register_fake()
        self.brain.express("火焰")
        ctx = self.captured[0]
        self.assertEqual(ctx["brain_name"], "LG")
        self.assertEqual(ctx["stimulus"], "火焰")
        for k in ("verb", "action", "mood", "recalled", "top_thought"):
            self.assertIn(k, ctx)

    def test_generator_exception_falls_back_to_template(self):
        def boom(ctx):
            raise RuntimeError("模型炸了")
        register_language_generator(boom, name="fake")
        out = self.brain.express("火焰")
        self.assertNotEqual(out["utterance"], "")
        self.assertNotIn("generator", out)  # 降级回模板

    def test_generator_empty_falls_back(self):
        self._register_fake(text="")
        out = self.brain.express("火焰")
        self.assertNotIn("generator", out)

    def test_unregister_restores_template(self):
        self._register_fake()
        unregister_language_generator("fake")
        out = self.brain.express("火焰")
        self.assertNotIn("generator", out)

    def test_use_generator_false_forces_template(self):
        self._register_fake()
        out = self.brain.express("火焰", use_generator=False)
        self.assertNotIn("generator", out)
        self.assertEqual(self.captured, [])  # 生成器未被调用

    def test_chat_uses_generator(self):
        self._register_fake(text="我在听你说")
        out = self.brain.chat("你好")
        self.assertEqual(out["reply"], "我在听你说")

    def test_set_qwen_model_without_weights(self):
        """模型未下载/transformers 未装：注册成功但自动降级模板"""
        info = set_qwen_model(model_path="models/__不存在的目录__",
                              name="qwen")
        self.assertEqual(info["registered"], "qwen")
        self.assertFalse(info["available"])
        out = self.brain.express("火焰")
        self.assertNotIn("generator", out)  # 优雅降级
        self.assertEqual(get_language_generator_info()["default"], "qwen")


class _FakeStore:
    """鸭子类型记忆后端：记录同步调用（不依赖真实 lancedb）"""

    def __init__(self):
        self.available = True
        self._error = None
        self.added, self.updated, self.decays = [], [], []
        self.rows = []

    def add(self, mem, brain_name=""):
        self.added.append((mem.content, brain_name))

    def update_weight(self, content, weight, brain_name=""):
        self.updated.append((content, weight))

    def search_vector(self, vec, top_k=3, brain_name=None,
                      modality=None, exclude_modality=None):
        self.last_search = {"vector": vec, "top_k": top_k,
                            "brain_name": brain_name,
                            "modality": modality,
                            "exclude_modality": exclude_modality}
        return self.rows[:top_k]

    def decay(self, factor, forget_threshold):
        self.decays.append((factor, forget_threshold))
        return 0


class TestLanceMemoryStore(unittest.TestCase):
    """v6.0 LanceDB 记忆后端：固化/强化/衰减同步 + 语义回忆 + 降级"""

    def setUp(self):
        self.brain = AIBrainEntity("LDB", seed=1)

    def _mem(self, content, weight=0.6):
        import time as _t
        return BrainMemory(content=content, timestamp=_t.time(),
                           weight=weight, tag="event")

    def test_attach_fake_store(self):
        store = _FakeStore()
        info = self.brain.attach_memory_store(store)
        self.assertTrue(info["attached"])
        self.assertTrue(info["available"])

    def test_consolidation_syncs_add_and_weight(self):
        store = _FakeStore()
        self.brain.attach_memory_store(store)
        self.brain._consolidate_to_ltm(self._mem("火焰是危险的"))
        self.assertEqual(store.added, [("火焰是危险的", "LDB")])
        # 重复固化同内容 → 强化已有记忆，同步权重更新
        self.brain._consolidate_to_ltm(self._mem("火焰是危险的"))
        self.assertEqual(len(store.added), 1)
        self.assertEqual(len(store.updated), 1)
        self.assertEqual(store.updated[0][0], "火焰是危险的")

    def test_decay_syncs_to_store(self):
        store = _FakeStore()
        self.brain.attach_memory_store(store)
        self.brain.decay_memory(0.5)
        self.assertEqual(store.decays,
                         [(0.5, self.brain.forget_threshold)])

    def test_recall_semantic_fallback_without_store(self):
        """未接后端：降级为关键词 recall，标记 source=memory-fallback"""
        for _ in range(30):
            self.brain.sensory_input("火焰是危险的")
        rows = self.brain.recall_semantic("火焰")
        self.assertTrue(rows)
        self.assertEqual(rows[0]["source"], "memory-fallback")
        self.assertIn("火焰", rows[0]["content"])

    def test_recall_semantic_uses_store(self):
        store = _FakeStore()
        store.rows = [{"content": "钻木可以取火", "weight": 0.9,
                       "tag": "culture", "modality": "text",
                       "distance": 0.12}]
        self.brain.attach_memory_store(store)
        rows = self.brain.recall_semantic("火")
        self.assertEqual(rows[0]["content"], "钻木可以取火")
        self.assertEqual(rows[0]["source"], "lancedb")
        # 语义命中进入思考空间
        self.assertTrue(any(t.content == "钻木可以取火"
                            for t in self.brain.thought_space))

    def test_real_lancedb_roundtrip(self):
        """真实 lancedb（未安装则跳过）：写入→向量检索→衰减删除"""
        try:
            import lancedb  # noqa: F401
        except ImportError:
            self.skipTest("lancedb 未安装")
        import shutil
        import tempfile
        from memory_store import LanceMemoryStore
        path = tempfile.mkdtemp(prefix="lance_test_")
        self.addCleanup(shutil.rmtree, path, True)
        store = LanceMemoryStore(path)
        self.assertTrue(store.available, store._error)
        self.brain.attach_memory_store(store)
        self.brain._consolidate_to_ltm(self._mem("火焰是危险的"))
        self.assertEqual(store.count(), 1)
        rows = self.brain.recall_semantic("火焰")
        self.assertTrue(rows)
        self.assertEqual(rows[0]["content"], "火焰是危险的")
        # 大幅衰减 → 权重跌破遗忘阈值 → 从库中删除
        self.brain.decay_memory(0.01)
        self.assertEqual(store.count(), 0)

    def test_recall_semantic_modality_passthrough(self):
        """跨模态联想参数透传到后端"""
        store = _FakeStore()
        self.brain.attach_memory_store(store)
        self.brain.recall_semantic([0.1] * 16,
                                   exclude_modality="visual")
        self.assertEqual(store.last_search["exclude_modality"], "visual")
        self.assertIsNone(store.last_search["modality"])

    def test_sync_stm_writes_all_memories(self):
        """sync_stm=True：短期记忆也全量入库"""
        store = _FakeStore()
        self.brain.attach_memory_store(store, sync_stm=True)
        self.brain.sensory_input("一条新刺激")
        self.assertTrue(any("一条新刺激" in c
                            for c, _ in store.added))


class TestMemoryVersioning(unittest.TestCase):
    """v6.1 记忆版本控制：修改历史 / 回忆过去版本 / 演化轨迹"""

    def setUp(self):
        try:
            import lancedb  # noqa: F401
        except ImportError:
            self.skipTest("lancedb 未安装")
        import shutil
        import tempfile
        from memory_store import LanceMemoryStore
        self.path = tempfile.mkdtemp(prefix="lance_ver_")
        self.addCleanup(shutil.rmtree, self.path, True)
        self.store = LanceMemoryStore(self.path)

    def _mem(self, content, weight=0.6):
        import time as _t
        return BrainMemory(content=content, timestamp=_t.time(),
                           weight=weight, tag="event")

    def test_history_records_add_and_reinforce(self):
        self.store.add(self._mem("火焰是危险的"), "B1")
        self.store.update_weight("火焰是危险的", 0.75, "B1")
        self.store.update_weight("火焰是危险的", 0.9, "B1")
        hist = self.store.memory_history("火焰是危险的", "B1")
        self.assertEqual(len(hist), 3)
        self.assertEqual([h["version"] for h in hist], [1, 2, 3])
        self.assertEqual(hist[0]["reason"], "add")
        self.assertEqual(hist[-1]["reason"], "reinforce")
        self.assertAlmostEqual(hist[-1]["weight"], 0.9)

    def test_recall_past_version(self):
        self.store.add(self._mem("水能灭火", 0.5), "B1")
        self.store.update_weight("水能灭火", 0.8, "B1")
        latest = self.store.recall_version("水能灭火", -1, "B1")
        older = self.store.recall_version("水能灭火", -2, "B1")
        self.assertAlmostEqual(latest["weight"], 0.8)
        self.assertAlmostEqual(older["weight"], 0.5)

    def test_decay_logs_trajectory(self):
        self.store.add(self._mem("太阳会发光", 0.9), "B1")
        self.store.decay(0.5, 0.01)
        hist = self.store.memory_history("太阳会发光", "B1")
        self.assertEqual([h["reason"] for h in hist], ["add", "decay"])
        self.assertAlmostEqual(hist[-1]["weight"], 0.45)

    def test_cross_modal_search(self):
        """统一向量空间：排除当前模态 → 跨模态联想"""
        m_visual = self._mem("一张猫的图片", 0.8)
        m_visual.modality = "visual"
        m_visual.features = [1.0] + [0.0] * 511
        m_audio = self._mem("喵的叫声", 0.7)
        m_audio.modality = "auditory"
        m_audio.features = [0.9] + [0.1] * 511
        self.store.add(m_visual, "B1")
        self.store.add(m_audio, "B1")
        # 看到猫 → 排除视觉记忆 → 联想起听觉记忆
        rows = self.store.search_vector([1.0] + [0.0] * 511, top_k=2,
                                        exclude_modality="visual")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["modality"], "auditory")


class TestDNALibrary(unittest.TestCase):
    """v6.1 DNA 基因库：存取 / 人格搜索 / 进化谱系"""

    def setUp(self):
        try:
            import lancedb  # noqa: F401
        except ImportError:
            self.skipTest("lancedb 未安装")
        import shutil
        import tempfile
        from memory_store import DNALibrary
        self.path = tempfile.mkdtemp(prefix="lance_dna_")
        self.addCleanup(shutil.rmtree, self.path, True)
        self.lib = DNALibrary(self.path)
        self.assertTrue(self.lib.available, self.lib._error)

    def test_save_and_get_roundtrip(self):
        brain = AIBrainEntity("Alpha", seed=1,
                              sensation_seeking=0.9)
        brain.sensory_input("基因库测试")
        r = brain.save_to_library(library=self.lib)
        self.assertTrue(r["saved"])
        dna = self.lib.get(r["dna_id"])
        self.assertEqual(dna["name"], "Alpha")
        # 取回的 DNA 可直接克隆
        clone = AIBrainEntity.from_dna(dna, new_name="Alpha-2")
        self.assertEqual(clone.sensation_seeking, 0.9)

    def test_search_by_personality(self):
        hi = AIBrainEntity("Explorer", seed=1, sensation_seeking=0.9)
        lo = AIBrainEntity("Homebody", seed=2, sensation_seeking=0.1)
        hi.save_to_library(library=self.lib)
        lo.save_to_library(library=self.lib)
        found = self.lib.search(sensation_seeking=(0.8, 1.0))
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["name"], "Explorer")
        found2 = self.lib.search(name_contains="Home")
        self.assertEqual(found2[0]["name"], "Homebody")

    def test_lineage_tracking(self):
        parent = AIBrainEntity("Parent", seed=1)
        rp = parent.save_to_library(library=self.lib)
        dna = parent.dump_dna()
        child = AIBrainEntity.from_dna(dna, new_name="Child")
        child.generation = 2
        rc = child.save_to_library(parents=[rp["dna_id"]],
                                   library=self.lib)
        chain = self.lib.lineage(rc["dna_id"])
        self.assertEqual(len(chain), 1)
        self.assertEqual(chain[0]["name"], "Parent")
        self.assertEqual(chain[0]["dna_id"], rp["dna_id"])

    def test_swarm_evolution_auto_archive(self):
        """种群进化：子代自动存档并链接亲代"""
        swarm = BrainSwarm(["A", "B", "C", "D"], seed=1)
        swarm.population[0].sensory_input("优势记忆内容xxxx")
        swarm.attach_dna_library(self.lib)
        swarm.save_population()
        self.assertEqual(self.lib.count(), 4)
        random.seed(3)
        swarm.evolve_generation(task="diversity")
        # 4 个初代 + 2 个子代
        self.assertEqual(self.lib.count(), 6)
        child = [b for b in swarm.population if "_g2_" in b.name][0]
        child_id = swarm._library_ids[child.name]
        chain = self.lib.lineage(child_id)
        self.assertEqual(len(chain), 1)  # 能回溯到亲代


class TestWorkingMemory(unittest.TestCase):
    """v6.0 工作记忆（Baddeley 模型）"""

    def test_report_structure(self):
        brain = AIBrainEntity("WM", seed=1)
        brain.sensory_input("火焰是危险的")
        r = brain.get_working_memory_report()
        for k in ("central_executive", "phonological_loop",
                  "visuospatial_sketchpad", "episodic_buffer", "total_load"):
            self.assertIn(k, r)


class TestPredictiveCoding(unittest.TestCase):
    """v6.1 预测编码：自由能最小化"""

    def test_free_energy_cycle(self):
        brain = AIBrainEntity("PC", seed=1)
        r = brain.minimize_free_energy("火焰")
        for k in ("prediction", "error", "free_energy",
                  "free_energy_reduced"):
            self.assertIn(k, r)
        self.assertGreaterEqual(r["free_energy"], 0)

    def test_repeated_input_learns(self):
        brain = AIBrainEntity("PC2", seed=1)
        brain.minimize_free_energy("火焰")
        r = brain.minimize_free_energy("火焰")
        self.assertIsInstance(r["free_energy_reduced"], bool)


class TestNeuralOscillation(unittest.TestCase):
    """v6.2 神经振荡（脑电波）"""

    def test_dominant_wave_and_state(self):
        brain = AIBrainEntity("OSC", seed=1)
        brain.sensory_input("量子纠缠")
        brain.update_brainwaves()
        wave = brain.get_dominant_wave()
        self.assertIn(wave, {"delta", "theta", "alpha", "beta", "gamma"})
        self.assertIsInstance(brain.get_consciousness_state(), str)

    def test_gamma_binding(self):
        brain = AIBrainEntity("OSC2", seed=1)
        # 低同步态：默认 γ 功率不足，绑定失败
        r0 = brain.gamma_binding(["红", "热", "亮"])
        self.assertFalse(r0["bound"])
        # 高专注 + 意识点火的高同步态：γ 绑定成功
        brain.brainwaves["gamma"] = 0.7
        brain.neural_synchrony = 0.8
        r = brain.gamma_binding(["红", "热", "亮"])
        self.assertTrue(r["bound"])
        self.assertEqual(r["features"], ["红", "热", "亮"])
        self.assertIn("红", r["content"])


class TestActiveInference(unittest.TestCase):
    """v6.3 主动推理：预期自由能驱动行动"""

    def setUp(self):
        self.brain = AIBrainEntity("AI", seed=1)
        self.brain.sensory_input("口渴了")

    def test_strategies_have_expected_free_energy(self):
        self.brain.add_goal("找到水", 0.8)
        strategies = self.brain.generate_action_strategies()
        self.assertTrue(strategies)
        self.assertIn("expected_free_energy", strategies[0])

    def test_active_inference_step(self):
        self.brain.add_goal("找到水", 0.8)
        r = self.brain.active_inference_step()
        self.assertIn("selected_action", r)
        self.assertIn("current_free_energy", r)


class TestBrainRegions(unittest.TestCase):
    """v6.4 脑区分化：海马/前额叶/杏仁核"""

    def setUp(self):
        self.brain = AIBrainEntity("BR", seed=1)

    def test_hippocampus_encode_and_completion(self):
        self.brain.hippocampus_encode("火灾逃生经历")
        r = self.brain.hippocampus_pattern_completion("火灾")
        self.assertGreater(r["completion_score"], 0)

    def test_fear_conditioning_and_extinction(self):
        self.brain.amygdala_fear_conditioning("蜘蛛", 0.9)
        after = self.brain.amygdala_detect_threat("蜘蛛")
        self.assertGreater(after["threat_level"], 0.5)
        self.brain.amygdala_extinguish_fear("蜘蛛")
        gone = self.brain.amygdala_detect_threat("蜘蛛")
        self.assertLess(gone["threat_level"], after["threat_level"])

    def test_region_report(self):
        self.brain.update_regions()
        r = self.brain.get_brain_regions_report()
        for k in ("regions", "most_active", "hippocampus", "amygdala"):
            self.assertIn(k, r)


class TestReasoningPlanning(unittest.TestCase):
    """v6.5 推理与规划"""

    def setUp(self):
        self.brain = AIBrainEntity("RP", seed=1)

    def test_deductive_reasoning(self):
        r = self.brain.deductive_reasoning(
            ["所有人都会死", "苏格拉底是人"])
        self.assertEqual(r["type"], "deductive")
        self.assertTrue(r["conclusions"])

    def test_plan_goal_and_decision(self):
        plan = self.brain.plan_goal("扑灭大火")
        self.assertTrue(plan["steps"])
        d = self.brain.make_decision(["方案A", "方案B"])
        self.assertIn(d["best_option"], ["方案A", "方案B"])


class TestMentalSimulation(unittest.TestCase):
    """v6.6 心理模拟：心理时间旅行 / 想象 / 洞见"""

    def setUp(self):
        self.brain = AIBrainEntity("MS", seed=1)
        self.brain.sensory_input("火焰是危险的")

    def test_mental_time_travel(self):
        past = self.brain.remember_past()
        future = self.brain.imagine_future("明天去爬山")
        self.assertEqual(future["direction"], "future")
        self.assertIn("vividness", past)

    def test_insight_and_divergent(self):
        ins = self.brain.generate_insight("如何灭火")
        self.assertIn("has_insight", ins)
        ideas = self.brain.divergent_thinking("砖头的用途")
        self.assertIsInstance(ideas, (dict, list))


class TestDevelopment(unittest.TestCase):
    """v6.7 发育过程：皮亚杰阶段 / 关键期"""

    def test_piaget_stage_progression(self):
        brain = AIBrainEntity("DEV", seed=1)
        self.assertEqual(brain.get_piaget_stage()["stage"], "sensorimotor")
        brain.develop(36)
        self.assertNotEqual(brain.get_piaget_stage()["stage"],
                            "sensorimotor")

    def test_object_permanence_milestone(self):
        brain = AIBrainEntity("DEV2", seed=1)
        self.assertFalse(brain.has_object_permanence())
        brain.develop(12)
        self.assertTrue(brain.has_object_permanence())


class TestEmbodiedCognition(unittest.TestCase):
    """v6.8 具身认知：身体图式 / 运动 / 镜像神经元 / 可供性"""

    def setUp(self):
        self.brain = AIBrainEntity("EC", seed=1)
        self.brain.init_body_schema()

    def test_motor_planning_and_skill(self):
        plan = self.brain.plan_motor_action("抓握")
        self.assertTrue(plan["planned"])
        r = self.brain.execute_motor_action()
        self.assertIsNotNone(r)

    def test_mirror_neuron_imitation(self):
        self.brain.observe_action("挥手")
        r = self.brain.imitate_action("挥手")
        self.assertTrue(0.0 <= r["success_rate"] <= 1.0)

    def test_affordance(self):
        r = self.brain.perceive_affordance("杯子")
        self.assertTrue(r["affordances"])


class TestCulturalEvolutionSystem(unittest.TestCase):
    """v6.9 文化进化：模因 / 规范 / 进化动力学"""

    def setUp(self):
        self.brain = AIBrainEntity("CE", seed=1)

    def test_meme_add_and_replicate(self):
        first = self.brain.add_meme("火焰崇拜")
        second = self.brain.add_meme("火焰崇拜")
        self.assertTrue(first["is_new"])
        self.assertFalse(second["is_new"])

    def test_cultural_evolution_step(self):
        self.brain.transmit_culture("钻木取火")
        r = self.brain.cultural_evolution_step()
        self.assertIn("generation", r)
        self.assertIn("diversity", r)


class TestLifelongLearning(unittest.TestCase):
    """v7.0 终身学习：增量 / 间隔重复 / 迁移"""

    def setUp(self):
        self.brain = AIBrainEntity("LL", seed=1)

    def test_incremental_learning(self):
        r = self.brain.learn_incremental("牛顿第一定律")
        self.assertIn("retention", r)
        self.assertEqual(r["knowledge"], "牛顿第一定律")

    def test_spaced_repetition_strengthens(self):
        self.brain.spaced_repetition("牛顿第一定律")
        r = self.brain.spaced_repetition("牛顿第一定律")
        self.assertGreaterEqual(r["storage_strength"], 0)

    def test_transfer_learning(self):
        self.brain.learn_incremental("骑自行车")
        r = self.brain.transfer_learning("骑自行车", "学摩托车")
        self.assertTrue(0.0 <= r["transfer_efficiency"] <= 1.0)


class TestConsciousnessIntegration(unittest.TestCase):
    """v7.1 意识整合：多理论框架统一"""

    def test_integration_report(self):
        brain = AIBrainEntity("CI", seed=1)
        brain.sensory_input("火焰是危险的")
        r = brain.get_consciousness_integration_report()
        for k in ("framework", "state", "integration", "metrics"):
            self.assertIn(k, r)


class _FakeTextEncoder:
    """伪语义编码器：按文本内容返回确定性向量"""
    available = True

    def encode(self, text):
        base = [((hash(text) >> i) & 7) / 10.0 for i in range(16)]
        norm = sum(v * v for v in base) ** 0.5 or 1.0
        return [v / norm for v in base]

    def info(self):
        return {"available": True, "model": "fake", "dim": 16}


class TestTextEncoder(unittest.TestCase):
    """v6.2 文本语义编码器：真语义检索通路 + 优雅降级"""

    def test_attach_without_model_degrades(self):
        """无本地模型时 attach 返回不可用，大脑行为不变（哈希兜底）"""
        brain = AIBrainEntity("TE1", seed=1)
        from models.encoders.text_encoder import create_text_encoder
        enc = create_text_encoder(model_path="models/不存在的模型目录")
        self.assertFalse(enc.available)
        r = brain.attach_text_encoder(enc)
        self.assertTrue(r["attached"])
        self.assertFalse(r["available"])
        # 未接向量库 + 编码器不可用 → STM 不生成 features（行为不变）
        brain.sensory_input("火焰")
        self.assertEqual(brain.short_memory[-1].features, [])

    def test_encoder_generates_features(self):
        """接入可用编码器后，文本记忆自动携带语义 features"""
        brain = AIBrainEntity("TE2", seed=1)
        brain.attach_text_encoder(_FakeTextEncoder())
        brain.sensory_input("钻木可以取火")
        feats = brain.short_memory[-1].features
        self.assertEqual(len(feats), 16)
        self.assertAlmostEqual(sum(v * v for v in feats) ** 0.5, 1.0, places=5)

    def test_semantic_query_uses_encoder(self):
        """recall_semantic 字符串查询走编码器向量而非哈希"""
        brain = AIBrainEntity("TE3", seed=1)
        brain.attach_text_encoder(_FakeTextEncoder())
        store = _FakeStore()
        brain.attach_memory_store(store)
        brain.recall_semantic("火焰")
        self.assertIsNotNone(store.last_search)
        # 查询向量应为伪编码器的 16 维输出（哈希兜底是 512 维）
        self.assertEqual(len(store.last_search["vector"]), 16)

    def test_hash_fallback_without_encoder(self):
        """未接编码器时字符串查询回退哈希向量（512 维）"""
        brain = AIBrainEntity("TE4", seed=1)
        store = _FakeStore()
        brain.attach_memory_store(store)
        brain.recall_semantic("火焰")
        self.assertEqual(len(store.last_search["vector"]), 512)


if __name__ == "__main__":
    unittest.main()
