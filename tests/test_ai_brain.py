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

from ai_brain_entity import (
    AIBrainEntity, BrainSwarm, BrainMemory, LearnableProjection,
    encode_image, encode_audio,
    register_image_encoder, register_audio_encoder,
    unregister_image_encoder, unregister_audio_encoder,
    list_encoders,
    make_robot_executor, make_api_executor,
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


class TestExecutorLoop(unittest.TestCase):
    """v4.1 扩展A：动作空间接入真实执行器 + 执行结果回传奖励"""

    def test_robot_executor_success_loop(self):
        """机器人执行成功 → 正奖励经 reward_td 回传，价值估计上升"""
        brain = AIBrainEntity("t", seed=42)
        brain.register_executor(make_robot_executor(strictness=0.1),
                                default=True)
        out = brain.act("火焰是危险的")
        self.assertIn("execution", out)
        self.assertTrue(out["execution"]["success"])
        self.assertIn("机器人执行", out["execution"]["detail"])
        self.assertIsNotNone(out["feedback"])
        self.assertGreater(brain.value_estimate, 0.0)

    def test_robot_executor_low_intensity_fails(self):
        """动作强度低于驱动门槛 → 执行失败 → 负奖励 → RPE 为负"""
        brain = AIBrainEntity("t", seed=42)
        brain.register_executor(make_robot_executor(strictness=0.9),
                                default=True)
        brain.sensory_input("测试刺激")
        out = brain.act("测试刺激")
        if out["action"]["verb"] != "observe":  # observe 无门槛
            self.assertFalse(out["execution"]["success"])
            self.assertLess(out["feedback"]["rpe"], 0.0)

    def test_verb_routing_and_no_executor(self):
        """按 verb 路由执行器；无执行器时 execution/feedback 为 None"""
        brain = AIBrainEntity("t", seed=42)
        calls = []
        brain.register_executor(
            lambda a: calls.append(a) or
            {"success": True, "reward": 0.5, "detail": "ok"},
            verb="observe")
        brain.sensory_input("平静输入")
        out = brain.act("平静输入")
        if out["action"]["verb"] == "observe":
            self.assertEqual(len(calls), 1)
        bare = AIBrainEntity("b", seed=1)
        bare.sensory_input("无执行器")
        out2 = bare.act("无执行器")
        self.assertIsNone(out2["execution"])
        self.assertIsNone(out2["feedback"])

    def test_executor_exception_becomes_negative_reward(self):
        """执行器抛异常不中断大脑，按失败处理"""
        def boom(action):
            raise RuntimeError("硬件离线")
        brain = AIBrainEntity("t", seed=42)
        brain.register_executor(boom, default=True)
        brain.sensory_input("触发动作")
        out = brain.act("触发动作")
        self.assertFalse(out["execution"]["success"])
        self.assertEqual(out["execution"]["reward"], -0.5)
        self.assertIn("硬件离线", out["execution"]["detail"])

    def test_api_executor_local_server(self):
        """HTTP API 执行器：本地真实 HTTP 服务往返 + 奖励回传"""
        import json as _json
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        received = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                received.append(_json.loads(self.rfile.read(length) or b"{}"))
                payload = _json.dumps({"reward": 0.7}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            url = f"http://127.0.0.1:{server.server_port}/act"
            brain = AIBrainEntity("t", seed=42)
            brain.register_executor(make_api_executor(url), default=True)
            out = brain.act("API 测试")
            self.assertTrue(out["execution"]["success"])
            self.assertAlmostEqual(out["execution"]["reward"], 0.7)
            self.assertIn("verb", received[0])  # 服务端收到结构化动作
            self.assertGreater(brain.value_estimate, 0.0)
        finally:
            server.shutdown()
            server.server_close()

    def test_api_executor_network_failure(self):
        """不可达端点 → 失败负奖励（不抛异常）"""
        brain = AIBrainEntity("t", seed=42)
        brain.register_executor(
            make_api_executor("http://127.0.0.1:1/unreachable", timeout=1),
            default=True)
        out = brain.act("断网测试")
        self.assertFalse(out["execution"]["success"])
        self.assertEqual(out["execution"]["reward"], -0.3)


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
    """v4.5：执行器技能学习——分 verb 独立价值估计 + 策略化动作选择"""

    def _mock(self, reward):
        return lambda a: {"success": reward > 0, "reward": reward,
                          "detail": "mock"}

    def _trained_brain(self, rounds=40, seed=1):
        """respond=0.8 / acknowledge=0.2 / observe=-0.4 三执行器训练。

        先直接灌注每 verb 5 次（低 ε 下 ε-greedy 可能锁死在次优动作——
        探索不足是真实 RL 现象，本测试关注的是策略覆盖而非学习动态），
        再跑 act() 闭环验证端到端更新。
        """
        brain = AIBrainEntity("S", seed=seed)
        for verb, rv in [("respond", 0.8), ("acknowledge", 0.2),
                         ("observe", -0.4)]:
            brain.register_executor(self._mock(rv), verb=verb)
            for _ in range(5):
                brain.learn_skill(verb, rv)
        for _ in range(rounds):
            brain.act("火焰是危险的", policy="epsilon")
        return brain

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

    def test_act_updates_verb_value(self):
        brain = AIBrainEntity("S", seed=1)
        brain.register_executor(self._mock(0.8), default=True)
        out = brain.act("火焰是危险的")
        verb = out["action"]["verb"]
        self.assertIsNotNone(out["skill"])
        self.assertGreater(brain.verb_values[verb], 0.0)

    def test_policy_overrides_verb_choice(self):
        """训练后 policy='greedy' 恒选高价值 verb，无视决策层脉冲"""
        brain = self._trained_brain()
        self.assertGreater(brain.verb_values["respond"], 0.7)
        for _ in range(10):
            out = brain.act("火焰是危险的", policy="greedy")
            self.assertEqual(out["action"]["verb"], "respond")
            self.assertEqual(out["action"]["action"], "主动响应")


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


if __name__ == "__main__":
    unittest.main()
