"""
导出100万参数类脑模型

流程: 进化优化 → 导出.npz → 加载验证 → 推理测试
"""
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from brain import (
    PredictiveCodingNetwork, EmotionalCore, MotivationSystem,
    ConsciousnessSystem, ThoughtSystem, MemorySystem, MockLanguageModel,
    DriveType, GeneticAlgorithm, HyperParams,
    flatten_pc_weights, unflatten_pc_weights,
    export_model, load_model, restore_modules, count_parameters,
)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "brain_v3.1_1m.npz")


def quick_evolve(n_gen=5, pop_size=8):
    """快速进化得到一个优化后的模型"""
    print("=" * 60)
    print("  阶段1: 进化优化")
    print("=" * 60)

    pc = PredictiveCodingNetwork([512, 650, 256])
    weight_dim = flatten_pc_weights(pc).shape[0]

    ga = GeneticAlgorithm(
        population_size=pop_size,
        weight_dim=weight_dim,
        weight_init_scale=0.05,
        tournament_k=3,
        elite_count=2,
        crossover_rate=0.7,
        mutation_rate=0.3,
        weight_mut_rate=0.03,
        weight_mut_sigma=0.015,
        hyper_mut_rate=0.3,
        seed=42,
    )

    eval_inputs = [
        ("你好", "greeting"), ("今天很开心", "positive"),
        ("为什么天是蓝的？", "curiosity"), ("危险！", "threat"),
        ("谢谢你", "gratitude"),
    ]

    for gen in range(n_gen):
        for ind in ga.population:
            pc_eval = PredictiveCodingNetwork([512, 650, 256])
            unflatten_pc_weights(pc_eval, ind.weights)
            emo = EmotionalCore()
            mot = MotivationSystem()
            h = ind.hyper.values
            pc_eval.lr = h["pc_learning_rate"]
            emo.dopamine.baseline = h["dopamine_baseline"]
            emo.dopamine.learning_rate = h["dopamine_learning"]

            total_err = 0
            emo_score = 0
            for text, cat in eval_inputs:
                emb = MockLanguageModel().embed(text)
                padded = np.zeros(512)
                padded[:128] = emb
                r = pc_eval.step(padded, 1.0)
                total_err += r["mean_error"]
                emo.evaluate_stimulus(text)
                if cat in ("positive", "gratitude"):
                    emo_score += max(0, emo.state.valence)
                elif cat == "threat":
                    emo_score += max(0, emo.state.fear)
                mot.evaluate(text, r["mean_error"],
                             threat_detected=(cat == "threat"))
                emo.step(); mot.step()

            n = len(eval_inputs)
            ind.fitness = (
                0.4 * max(0, 1 - total_err / n * 2) +
                0.3 * emo_score / n +
                0.3 * min(1, mot.drives[DriveType.CURIOSITY].level)
            )

        stats = ga.evolve()
        print(f"  代{stats['generation']:>2}: "
              f"最佳={stats['best_fitness']:.4f} "
              f"平均={stats['avg_fitness']:.4f}")

    print(f"  进化完成, 最佳适应度={ga.best_genome.fitness:.4f}")
    return ga.best_genome


def main():
    # 阶段1: 进化
    best = quick_evolve(n_gen=5, pop_size=8)

    # 阶段2: 构建最佳模型并运行几步
    print("\n" + "=" * 60)
    print("  阶段2: 构建最佳模型")
    print("=" * 60)

    pc = PredictiveCodingNetwork([512, 650, 256])
    unflatten_pc_weights(pc, best.weights)

    emo = EmotionalCore()
    mot = MotivationSystem()
    con = ConsciousnessSystem()
    ts = ThoughtSystem(vector_dim=512)
    mem = MemorySystem()
    llm = MockLanguageModel()

    # 应用超参数
    h = best.hyper.values
    pc.lr = h["pc_learning_rate"]
    pc.layers[0].precision = h["pc_precision_input"]
    pc.layers[1].precision = h["pc_precision_mid"]
    pc.layers[2].precision = h["pc_precision_top"]
    emo.decay_rate = h["emotion_decay"]
    emo.dopamine.baseline = h["dopamine_baseline"]
    emo.dopamine.learning_rate = h["dopamine_learning"]
    mot.base_explore = h["explore_base"]
    con.workspace.noise_level = h["conscious_noise"]

    # 运行几步让模型进入工作状态
    print("  预热运行...")
    for text in ["你好", "今天很开心", "为什么天是蓝的？", "危险！", "谢谢"]:
        emb = llm.embed(text)
        padded = np.zeros(512)
        padded[:128] = emb
        pc.step(padded, 1.0)
        emo.evaluate_stimulus(text)
        mot.evaluate(text, pc.get_last_error(),
                     threat_detected="危险" in text, social_interaction=True)
        emo.dopamine.compute_rpe(0.5, "warmup")
        cands = con.build_candidates(user_input=text, emotion_state=emo.state,
                                     prediction_error=pc.get_last_error(),
                                     curiosity=0.5)
        if cands:
            con.workspace.compete(cands)
        ts.input_perceptual(padded, 0.7)
        mem.input_sensory(emb, emo.state.valence)
        mem.step(dopamine_level=emo.dopamine.current_dopamine)
        ts.step(); emo.step(); mot.step(); con.step()

    print(f"  参数量: {count_parameters(pc):,}")
    print(f"  预测误差: {pc.get_last_error():.4f}")
    print(f"  情绪: {emo.state.dominant()}, DA={emo.dopamine.current_dopamine:.3f}")

    # 阶段3: 导出
    print("\n" + "=" * 60)
    print("  阶段3: 导出模型")
    print("=" * 60)

    os.makedirs(MODEL_DIR, exist_ok=True)
    path = export_model(
        pc=pc,
        hyper=best.hyper,
        emotion=emo,
        motivation=mot,
        consciousness=con,
        thought=ts,
        memory=mem,
        filepath=MODEL_PATH,
        metadata={
            "name": "Brain Simulator v3.1",
            "description": "100万参数类脑认知模型",
            "fitness": float(best.fitness),
            "training": "5代遗传算法进化",
        },
    )

    # 阶段4: 加载验证
    print("\n" + "=" * 60)
    print("  阶段4: 加载验证")
    print("=" * 60)

    pc2, state = load_model(MODEL_PATH)

    # 恢复模块状态
    emo2 = EmotionalCore()
    mot2 = MotivationSystem()
    con2 = ConsciousnessSystem()
    restore_modules(state, emotion=emo2, motivation=mot2, consciousness=con2)

    # 验证权重一致性
    w1 = flatten_pc_weights(pc)
    w2 = flatten_pc_weights(pc2)
    weight_match = np.allclose(w1, w2)
    print(f"  权重一致性: {'PASS' if weight_match else 'FAIL'}")
    print(f"  权重差异: {np.max(np.abs(w1 - w2)):.2e}")

    # 验证状态恢复
    print(f"  情绪恢复: {emo2.state.dominant()} "
          f"DA={emo2.dopamine.current_dopamine:.3f} "
          f"(原始: {emo.state.dominant()} DA={emo.dopamine.current_dopamine:.3f})")
    print(f"  意识噪声: {con2.workspace.noise_level:.3f} "
          f"(原始: {con.workspace.noise_level:.3f})")

    if state.get("hyper"):
        print(f"  超参数: {len(state['hyper'].values)}个已保存")
    print(f"  元信息: {state['meta']['name']} v{state['meta']['version']}")
    print(f"  适应度: {state['meta'].get('fitness', 'N/A')}")

    # 阶段5: 加载后推理测试
    print("\n" + "=" * 60)
    print("  阶段5: 加载模型推理测试")
    print("=" * 60)

    llm2 = MockLanguageModel()
    test_inputs = ["你好", "我很开心", "危险！", "再见"]
    for text in test_inputs:
        emb = llm2.embed(text)
        padded = np.zeros(512)
        padded[:128] = emb
        r = pc2.step(padded, 1.0)
        emo2.evaluate_stimulus(text)
        temp, top_p = emo2.get_generation_params()
        resp = llm2.generate(text, temperature=temp, top_p=top_p)
        print(f"  {text:10s} → {resp:22s} "
              f"err={r['mean_error']:.4f} "
              f"{emo2.state.dominant()} DA={emo2.dopamine.current_dopamine:.3f}")
        emo2.step()

    print("\n" + "=" * 60)
    print(f"  模型导出完成: {MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
