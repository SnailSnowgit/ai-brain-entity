"""
Brain Simulator v3.0 — 100万参数类脑认知系统
8大核心模块联动运行
"""
import numpy as np
import time
from brain import (
    MockLanguageModel, MemorySystem, EmotionalCore,
    ConsciousnessSystem, MotivationSystem, MessageBus, Message, MessageType,
    PredictiveCodingNetwork, ThoughtSystem, DriveType,
)


class ConsciousLLM:
    """意识LLM — 整合全部8大模块"""

    def __init__(self):
        # 1. 基础LLM
        self.llm = MockLanguageModel()
        # 2. 类脑记忆
        self.memory = MemorySystem(
            sensory_buffer_size=200, stm_size=100, ltm_size=5000)
        # 3. 情绪核心
        self.emotion = EmotionalCore()
        # 4. 意识(GWT)
        self.consciousness = ConsciousnessSystem()
        # 5. 内在动机
        self.motivation = MotivationSystem()
        # 6. 通信总线
        self.bus = MessageBus()
        # 7. 预测编码(100万参数)
        self.predictor = PredictiveCodingNetwork([512, 650, 256])
        # 8. 思考系统
        self.thought = ThoughtSystem(space_capacity=30, vector_dim=512)

        # 注册总线
        self.bus.subscribe("memory", MessageType.MEMORY_RECALL,
                           lambda m: self._on_memory(m))
        self.bus.subscribe("emotion", MessageType.EMOTION_UPDATE,
                           lambda m: self._on_emotion(m))

        self.step_count = 0
        self.total_params = self._count_params()

    def _count_params(self):
        p = 0
        for layer in self.predictor.layers:
            p += layer.top_down_weights.size
            p += layer.bottom_up_weights.size
            p += layer.activation.size * 3
        p += 512 * 30  # thought space
        return p

    def _on_memory(self, msg):
        pass

    def _on_emotion(self, msg):
        pass

    def step(self, user_input: str = None) -> dict:
        """单步认知循环"""
        self.step_count += 1
        dt = 1.0

        # === 1. 感觉输入 ===
        if user_input:
            embedding = self.llm.embed(user_input)
            # 提升到512维(零填充)
            if embedding.shape[0] < 512:
                padded = np.zeros(512)
                padded[:embedding.shape[0]] = embedding
                embedding = padded
            self.thought.input_perceptual(embedding, strength=0.8)
        else:
            embedding = None

        # === 2. 预测编码 ===
        if embedding is not None:
            pc_result = self.predictor.step(embedding, dt=dt)
            pred_error = pc_result['mean_error']
        else:
            # 无输入时自由预测
            pred_error = self.predictor.get_last_error()

        # === 3. 情绪评估 ===
        if user_input:
            self.emotion.evaluate_stimulus(user_input)

        # === 4. 意识竞争(GWT) ===
        candidates = self.consciousness.build_candidates(
            user_input=user_input,
            emotion_state=self.emotion.state,
            prediction_error=pred_error,
            curiosity=self.motivation.drives[DriveType.CURIOSITY].level,
        )
        conscious = self.consciousness.workspace.compete(candidates)

        # === 5. 动机评估 ===
        mot_result = self.motivation.evaluate(
            user_input=user_input,
            prediction_error=pred_error,
            social_interaction=user_input is not None,
            threat_detected=any(w in (user_input or "") for w in
                                ["危险", "害怕", "恐惧", "救命"]),
        )

        # === 6. 多巴胺RPE ===
        rpe = self.emotion.dopamine.compute_rpe(
            mot_result['reward'], f"step_{self.step_count}")

        # === 7. 思考 ===
        if embedding is not None:
            self.thought.input_perceptual(embedding, conscious.salience)
            self.thought.input_emotional(
                np.random.randn(512) * 0.1 + self.emotion.state.arousal,
                strength=self.emotion.state.arousal)
        # 高好奇心/预测误差触发系统2
        if mot_result['curiosity'] > 0.5 or pred_error > 0.3:
            self.thought.activate_system2(steps=2)
        self.thought.step(dt=dt)

        # === 8. 记忆巩固 ===
        if embedding is not None:
            self.memory.input_sensory(
                embedding[:128],  # 记忆用128维
                emotional_valence=self.emotion.state.valence)
        self.memory.step(dt=dt, dopamine_level=self.emotion.dopamine.current_dopamine)

        # === 9. LLM生成(情绪调制) ===
        temp, top_p = self.emotion.get_generation_params()
        if user_input:
            response = self.llm.generate(
                user_input, temperature=temp, top_p=top_p)
        else:
            response = "(内心活动...)"

        # === 10. 总线广播 ===
        self.bus.publish(Message(
            sender="consciousness",
            msg_type=MessageType.CONSCIOUS_BROADCAST,
            content={"text": conscious.text, "source": conscious.source},
            priority=10,
        ))

        # === 11. 各模块步进 ===
        self.emotion.step(dt=dt)
        self.motivation.step(dt=dt)
        self.consciousness.step(dt=dt)

        return {
            'step': self.step_count,
            'input': user_input,
            'response': response,
            'emotion': self.emotion.state.dominant(),
            'valence': self.emotion.state.valence,
            'arousal': self.emotion.state.arousal,
            'dopamine': self.emotion.dopamine.current_dopamine,
            'cortisol': 0.15 + 0.5 * max(0, self.emotion.state.fear - 0.3),
            'pred_error': pred_error,
            'curiosity': mot_result['curiosity'],
            'reward': mot_result['reward'],
            'dominant_drive': mot_result['dominant_drive'],
            'conscious': conscious.text[:50],
            'conscious_source': conscious.source,
            'phi': self.consciousness.metrics.phi,
            'thought_type': self.thought.get_summary()['current_type'],
            'memory_stats': self.memory.get_stats(),
            'temperature': temp,
        }


def main():
    print("=" * 65)
    print("   Brain Simulator v3.0 — 100万参数类脑认知系统")
    print("=" * 65)

    brain = ConsciousLLM()
    print(f"   参数量: {brain.total_params:,}")
    print(f"   模块: LLM | 记忆 | 情绪 | 意识 | 动机 | 总线 | 预测编码 | 思考")
    print("=" * 65)

    # 演示对话
    dialogues = [
        "你好",
        "今天很开心",
        "为什么天空是蓝色的？",
        "我有点害怕",
        "危险！快逃！",
        "谢谢你",
        "我想学习新知识",
        "再见",
    ]

    print("\n" + "─" * 65)
    print(f"  {'步数':>4} {'输入':<16} {'情绪':<6} {'DA':>5} {'皮质醇':>6} "
          f"{'预测误差':>7} {'好奇心':>6} {'Φ':>5} {'思维':<6}")
    print("─" * 65)

    for msg in dialogues:
        r = brain.step(msg)
        print(f"  {r['step']:>4} {r['input']:<16} {r['emotion']:<6} "
              f"{r['dopamine']:.3f} {r['cortisol']:.3f}  "
              f"{r['pred_error']:.4f}  {r['curiosity']:.3f}  "
              f"{r['phi']:.3f} {r['thought_type']:<6}")
        print(f"       → {r['response']}")
        print(f"       意识[{r['conscious_source']}]: {r['conscious']}")
        print()

    # 自由思考(无输入)
    print("─" * 65)
    print("  自由思考阶段(无外部输入)")
    print("─" * 65)
    for i in range(5):
        r = brain.step(None)
        print(f"  {r['step']:>4} {'(内省)':<16} {r['emotion']:<6} "
              f"{r['dopamine']:.3f} {r['cortisol']:.3f}  "
              f"{r['pred_error']:.4f}  {r['curiosity']:.3f}  "
              f"{r['phi']:.3f} {r['thought_type']:<6}")
        print(f"       意识: {r['conscious']}")
        time.sleep(0.1)

    # 最终状态
    print("\n" + "=" * 65)
    print("  系统状态汇总")
    print("=" * 65)
    ms = brain.memory.get_stats()
    print(f"  总步数: {brain.step_count}")
    print(f"  记忆: 感觉缓冲={ms['sensory_buffer_count']} "
          f"短期={ms['stm_count']} 长期={ms['ltm_count']}")
    print(f"  多巴胺基线: {brain.emotion.dopamine.baseline:.3f}")
    print(f"  意识水平: {brain.consciousness.determine_level(brain.emotion.state.arousal).value}")
    print(f"  Φ(信息整合): {brain.consciousness.metrics.phi:.4f}")
    print(f"  意识广播次数: {brain.consciousness.metrics.broadcast_count}")
    print(f"  预测编码最终误差: {brain.predictor.get_last_error():.4f}")
    print(f"  总线消息数: {brain.bus.get_stats()['total_messages']}")
    print(f"  思维流长度: {len(brain.thought.stream.stream)}")
    print(f"  参数量: {brain.total_params:,}")
    print("=" * 65)


if __name__ == "__main__":
    main()
