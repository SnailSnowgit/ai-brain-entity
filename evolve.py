"""
进化运行 — 遗传算法优化100万参数类脑系统

适应度 = 预测准确性 + 情绪响应适当性 + 多巴胺可塑性 + 记忆效率
"""
import numpy as np
import time
from brain import (
    PredictiveCodingNetwork, EmotionalCore,
    MotivationSystem, ConsciousnessSystem, ThoughtSystem, MemorySystem,
    MockLanguageModel, DriveType,
    GeneticAlgorithm, Genome, HyperParams,
    flatten_pc_weights, unflatten_pc_weights,
)


# 评估用的测试输入
EVAL_INPUTS = [
    ("你好",            "greeting"),
    ("今天很开心",       "positive"),
    ("为什么天空是蓝的？", "curiosity"),
    ("我有点害怕",       "fear"),
    ("危险！快逃！",     "threat"),
    ("谢谢你",          "gratitude"),
    ("我想学习",        "curiosity"),
    ("再见",            "greeting"),
]


def evaluate_genome(genome: Genome, n_eval_steps: int = 8) -> float:
    """
    评估个体适应度

    维度:
      1. 预测编码误差 (越低越好)
      2. 情绪响应适当性 (正面→正效价, 威胁→恐惧)
      3. 多巴胺RPE合理性 (奖励→正RPE, 威胁→负RPE)
      4. 记忆巩固 (有信息进入短期记忆)
      5. 意识水平 (保持清醒而非深睡)
    """
    # 构建系统
    pc = PredictiveCodingNetwork([512, 650, 256])
    unflatten_pc_weights(pc, genome.weights)

    emo = EmotionalCore()
    mot = MotivationSystem()
    con = ConsciousnessSystem()
    ts = ThoughtSystem(vector_dim=512)
    mem = MemorySystem(sensory_buffer_size=50, stm_size=30, ltm_size=100)
    llm = MockLanguageModel()

    # 应用超参数
    h = genome.hyper.values
    pc.lr = h["pc_learning_rate"]
    pc.layers[0].precision = h["pc_precision_input"]
    pc.layers[1].precision = h["pc_precision_mid"]
    pc.layers[2].precision = h["pc_precision_top"]
    emo.decay_rate = h["emotion_decay"]
    emo.dopamine.baseline = h["dopamine_baseline"]
    emo.dopamine.learning_rate = h["dopamine_learning"]
    mot.base_explore = h["explore_base"]
    con.workspace.noise_level = h["conscious_noise"]

    fitness = 0.0
    total_pred_error = 0.0
    emotion_appropriate = 0.0
    dopamine_appropriate = 0.0
    memory_score = 0.0
    consciousness_score = 0.0

    for i, (text, category) in enumerate(EVAL_INPUTS[:n_eval_steps]):
        emb = llm.embed(text)
        padded = np.zeros(512)
        padded[:128] = emb

        # 预测编码
        pc_result = pc.step(padded, dt=1.0)
        total_pred_error += pc_result["mean_error"]

        # 情绪
        emo.evaluate_stimulus(text)
        if category == "positive" or category == "gratitude":
            emotion_appropriate += max(0, emo.state.valence)
        elif category == "threat" or category == "fear":
            emotion_appropriate += max(0, emo.state.fear)
        elif category == "curiosity":
            emotion_appropriate += 0.3  # 中性偏好奇

        # 多巴胺RPE
        mot_result = mot.evaluate(
            user_input=text,
            prediction_error=pc_result["mean_error"],
            threat_detected=(category == "threat"),
            social_interaction=True,
        )
        rpe = emo.dopamine.compute_rpe(mot_result["reward"], f"eval_{i}")
        if category in ("positive", "gratitude"):
            dopamine_appropriate += max(0, rpe)
        elif category in ("threat", "fear"):
            dopamine_appropriate += max(0, -rpe)

        # 记忆
        mem.input_sensory(emb, emotional_valence=emo.state.valence)
        mem.step(dt=1.0, dopamine_level=emo.dopamine.current_dopamine)

        # 意识
        candidates = con.build_candidates(
            user_input=text,
            emotion_state=emo.state,
            prediction_error=pc_result["mean_error"],
            curiosity=mot.drives[DriveType.CURIOSITY].level,
        )
        if candidates:
            winner = con.workspace.compete(candidates)
            consciousness_score += winner.salience
        con.step()

        # 思考
        ts.input_perceptual(padded, strength=0.7)
        if category == "curiosity":
            ts.activate_system2(steps=2)
        ts.step()

        emo.step()
        mot.step()

    n = n_eval_steps

    # 1. 预测误差分 (误差越低分越高, 映射到0-1)
    avg_error = total_pred_error / n
    pred_score = max(0, 1.0 - avg_error * 2)

    # 2. 情绪适当性 (0-1)
    emotion_score = emotion_appropriate / n

    # 3. 多巴胺适当性 (0-1)
    dopamine_score = min(1.0, dopamine_appropriate / (n * 0.5))

    # 4. 记忆分
    ms = mem.get_stats()
    memory_score = min(1.0, ms["stm_count"] / (n * 0.5))

    # 5. 意识分
    consciousness_score = consciousness_score / n

    # 加权综合
    fitness = (
        0.30 * pred_score +
        0.20 * emotion_score +
        0.20 * dopamine_score +
        0.15 * memory_score +
        0.15 * consciousness_score
    )

    return float(fitness)


def run_evolution(pop_size: int = 10, n_generations: int = 10,
                  eval_steps: int = 8):
    """运行进化过程"""
    print("=" * 65)
    print("   遗传算法进化 — 100万参数类脑系统")
    print("=" * 65)

    # 计算权重维度
    pc = PredictiveCodingNetwork([512, 650, 256])
    weight_dim = flatten_pc_weights(pc).shape[0]
    print(f"   权重维度: {weight_dim:,}")
    print(f"   超参数: {len(HyperParams.DEFINITIONS)}个")
    print(f"   种群大小: {pop_size}")
    print(f"   进化代数: {n_generations}")
    print(f"   评估步数: {eval_steps}")
    print("=" * 65)

    # 初始化遗传算法
    ga = GeneticAlgorithm(
        population_size=pop_size,
        weight_dim=weight_dim,
        weight_init_scale=0.05,
        tournament_k=3,
        elite_count=2,
        crossover_rate=0.7,
        mutation_rate=0.3,
        weight_mut_rate=0.02,
        weight_mut_sigma=0.01,
        hyper_mut_rate=0.3,
        seed=42,
    )

    print(f"\n{'代数':>4} {'最佳':>8} {'平均':>8} {'最差':>8} "
          f"{'多样性':>10} {'耗时':>6}")
    print("-" * 65)

    overall_start = time.time()

    for gen in range(n_generations):
        gen_start = time.time()

        # 评估每个个体
        for i, individual in enumerate(ga.population):
            individual.fitness = evaluate_genome(individual, eval_steps)

        # 进化
        stats = ga.evolve()
        gen_time = time.time() - gen_start

        print(f"{stats['generation']:>4} "
              f"{stats['best_fitness']:>8.4f} "
              f"{stats['avg_fitness']:>8.4f} "
              f"{stats['worst_fitness']:>8.4f} "
              f"{ga.get_diversity():>10.6f} "
              f"{gen_time:>5.1f}s")

    total_time = time.time() - overall_start

    # 最终结果
    print("\n" + "=" * 65)
    print("   进化完成")
    print("=" * 65)
    print(f"   总耗时: {total_time:.1f}s")
    print(f"   初始最佳: {ga.best_fitness_history[0]:.4f}")
    print(f"   最终最佳: {ga.best_fitness_history[-1]:.4f}")
    improvement = (ga.best_fitness_history[-1] - ga.best_fitness_history[0])
    print(f"   提升: {improvement:+.4f} ({improvement/ga.best_fitness_history[0]*100:+.1f}%)")

    print(f"\n   最佳超参数:")
    for name, lo, hi, sigma in HyperParams.DEFINITIONS:
        val = ga.best_genome.hyper.get(name)
        default = (lo + hi) / 2
        marker = " *" if abs(val - default) > (hi - lo) * 0.15 else ""
        print(f"     {name:25s} = {val:.4f}  (默认{default:.4f}){marker}")

    # 用最佳个体运行演示
    print("\n" + "=" * 65)
    print("   最佳个体行为演示")
    print("=" * 65)

    best = ga.best_genome
    pc_best = PredictiveCodingNetwork([512, 650, 256])
    unflatten_pc_weights(pc_best, best.weights)

    emo = EmotionalCore()
    mot = MotivationSystem()
    con = ConsciousnessSystem()
    ts = ThoughtSystem(vector_dim=512)
    mem = MemorySystem()
    llm = MockLanguageModel()
    apply_hyperparams_to_system(pc_best, emo, mot, con, best.hyper)

    for text, category in EVAL_INPUTS:
        emb = llm.embed(text)
        padded = np.zeros(512)
        padded[:128] = emb

        pc_r = pc_best.step(padded, 1.0)
        emo.evaluate_stimulus(text)
        mot_r = mot.evaluate(text, pc_r["mean_error"],
                             threat_detected=(category == "threat"),
                             social_interaction=True)
        rpe = emo.dopamine.compute_rpe(mot_r["reward"], "demo")
        cands = con.build_candidates(user_input=text, emotion_state=emo.state,
                                     prediction_error=pc_r["mean_error"],
                                     curiosity=mot.drives[DriveType.CURIOSITY].level)
        winner = con.workspace.compete(cands) if cands else None
        temp, top_p = emo.get_generation_params()
        resp = llm.generate(text, temperature=temp, top_p=top_p)

        mem.input_sensory(emb, emo.state.valence)
        mem.step(dopamine_level=emo.dopamine.current_dopamine)
        ts.input_perceptual(padded, 0.7)
        ts.step()
        emo.step(); mot.step(); con.step()

        print(f"  {text:16s} → {resp:22s} "
              f"{emo.state.dominant():4s} DA={emo.dopamine.current_dopamine:.3f} "
              f"err={pc_r['mean_error']:.4f}")

    ms = mem.get_stats()
    print(f"\n  记忆: 短期={ms['stm_count']} 长期={ms['ltm_count']}")
    print(f"  适应度: {best.fitness:.4f}")

    return ga


def apply_hyperparams_to_system(pc, emo, mot, con, hyper):
    """应用超参数到各模块"""
    h = hyper.values
    pc.lr = h["pc_learning_rate"]
    pc.layers[0].precision = h["pc_precision_input"]
    pc.layers[1].precision = h["pc_precision_mid"]
    pc.layers[2].precision = h["pc_precision_top"]
    emo.decay_rate = h["emotion_decay"]
    emo.dopamine.baseline = h["dopamine_baseline"]
    emo.dopamine.learning_rate = h["dopamine_learning"]
    mot.base_explore = h["explore_base"]
    con.workspace.noise_level = h["conscious_noise"]


if __name__ == "__main__":
    run_evolution(pop_size=10, n_generations=10, eval_steps=8)
