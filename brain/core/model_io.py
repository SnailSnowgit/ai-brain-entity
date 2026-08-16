"""
模型导出/加载 — 保存和恢复类脑模型(v4.0, 18模块)

导出内容:
  - 预测编码网络权重 (998,400参数)
  - 基底神经节权重 (D1/D2/Critic)
  - 小脑权重 (颗粒/浦肯野/时序)
  - 强化学习权重 (Actor/Critic)
  - 15个进化超参数
  - 各模块状态
  - 模型元信息

格式: .npz (NumPy压缩)
"""

import numpy as np
import json
import time
import os
from typing import Dict, Any, Optional, List, Tuple

from .predictive_coding import PredictiveCodingNetwork
from .emotion import EmotionalCore
from .motivation import MotivationSystem
from .consciousness import ConsciousnessSystem
from .thought import ThoughtSystem
from .memory import MemorySystem
from .evolution import HyperParams

MODEL_VERSION = "4.0"
DEFAULT_LAYER_SIZES = [512, 650, 256]


def count_parameters(pc: PredictiveCodingNetwork = None,
                     bg=None, cer=None, rl=None) -> int:
    """统计总参数量"""
    total = 0
    if pc is not None:
        for layer in pc.layers:
            total += layer.top_down_weights.size
            total += layer.bottom_up_weights.size
            total += layer.activation.size
            total += layer.prediction.size
            total += layer.error.size
    if bg is not None:
        total += bg.d1_weights.size + bg.d2_weights.size
        total += bg.critic_weights.size
    if cer is not None:
        total += cer.granular_weights.size + cer.purkinje_weights.size
        total += cer.timing_weights.size
    if rl is not None:
        total += rl.critic.weights.size + rl.actor.weights.size
    return total


def export_model(
    pc: PredictiveCodingNetwork,
    hyper: HyperParams = None,
    emotion: EmotionalCore = None,
    motivation: MotivationSystem = None,
    consciousness: ConsciousnessSystem = None,
    thought: ThoughtSystem = None,
    memory: MemorySystem = None,
    # v4.0 新模块
    basal_ganglia=None,
    attention=None,
    sleep=None,
    homeostasis=None,
    default_mode=None,
    metacognition=None,
    cerebellum=None,
    reinforcement=None,
    goals=None,
    emotion_regulation=None,
    filepath: str = "brain_model.npz",
    metadata: Dict[str, Any] = None,
) -> str:
    """导出完整模型到 .npz 文件"""
    save_dict = {}

    # === 元信息 ===
    meta = {
        "version": MODEL_VERSION,
        "timestamp": time.time(),
        "time_str": time.strftime("%Y-%m-%d %H:%M:%S"),
        "layer_sizes": [layer.size for layer in pc.layers],
        "total_params": count_parameters(pc, basal_ganglia, cerebellum, reinforcement),
        "pc_learning_rate": float(pc.lr),
        "pc_time_constant": float(pc.time_constant),
        "modules": [
            "language_model", "memory", "emotion", "consciousness",
            "motivation", "bus", "predictive_coding", "thought",
            "basal_ganglia", "attention", "sleep", "homeostasis",
            "default_mode", "metacognition", "cerebellum",
            "reinforcement", "goals", "emotion_regulation",
        ],
    }
    if metadata:
        for k, v in metadata.items():
            if k != "modules":
                meta[k] = v
    save_dict["__meta__"] = np.array([json.dumps(meta, ensure_ascii=False)])

    # === 预测编码权重 ===
    weight_parts = []
    weight_shapes = []
    weight_names = []
    for i, layer in enumerate(pc.layers):
        for wname in ["top_down_weights", "bottom_up_weights"]:
            w = getattr(layer, wname)
            if w.size > 1:
                weight_parts.append(w.flatten())
                weight_shapes.append(list(w.shape))
                weight_names.append(f"pc_layer{i}_{wname}")
        save_dict[f"pc_layer{i}_activation"] = layer.activation.copy()
        save_dict[f"pc_layer{i}_prediction"] = layer.prediction.copy()
        save_dict[f"pc_layer{i}_error"] = layer.error.copy()
        save_dict[f"pc_layer{i}_precision"] = np.array([layer.precision])

    save_dict["pc_weights"] = np.concatenate(weight_parts)
    save_dict["__pc_weight_shapes__"] = np.array([
        json.dumps({"names": weight_names, "shapes": weight_shapes})
    ])

    # === 基底神经节权重 ===
    if basal_ganglia is not None:
        save_dict["bg_d1_weights"] = basal_ganglia.d1_weights.copy()
        save_dict["bg_d2_weights"] = basal_ganglia.d2_weights.copy()
        save_dict["bg_critic_weights"] = basal_ganglia.critic_weights.copy()
        bg_actions = {}
        for aid, a in basal_ganglia.actions.items():
            bg_actions[aid] = {
                "name": a.name,
                "action_type": a.action_type.name,
                "q_value": a.q_value,
                "habit_strength": a.habit_strength,
                "execution_count": a.execution_count,
                "success_count": a.success_count,
            }
        save_dict["__bg_actions__"] = np.array([json.dumps(bg_actions, ensure_ascii=False)])
        save_dict["__bg_state__"] = np.array([json.dumps({
            "total_selections": basal_ganglia.total_selections,
            "habit_selections": basal_ganglia.habit_selections,
            "lr": basal_ganglia.lr,
            "habit_threshold": basal_ganglia.habit_threshold,
        })])

    # === 小脑权重 ===
    if cerebellum is not None:
        save_dict["cer_granular_weights"] = cerebellum.granular_weights.copy()
        save_dict["cer_purkinje_weights"] = cerebellum.purkinje_weights.copy()
        save_dict["cer_timing_weights"] = cerebellum.timing_weights.copy()
        save_dict["__cer_state__"] = np.array([json.dumps({
            "state_dim": cerebellum.state_dim,
            "command_dim": cerebellum.command_dim,
            "lr": cerebellum.lr,
            "correction_gain": cerebellum.correction_gain,
            "sequences": {
                sid: {
                    "name": s.name,
                    "repetitions": s.repetitions,
                    "success_count": s.success_count,
                    "automaticity": s.automaticity,
                }
                for sid, s in cerebellum.sequences.items()
            },
        }, ensure_ascii=False)])

    # === 强化学习权重 ===
    if reinforcement is not None:
        save_dict["rl_critic_weights"] = reinforcement.critic.weights.copy()
        save_dict["rl_actor_weights"] = reinforcement.actor.weights.copy()
        save_dict["__rl_state__"] = np.array([json.dumps({
            "state_dim": reinforcement.critic.state_dim,
            "n_actions": reinforcement.actor.n_actions,
            "gamma": reinforcement.gamma,
            "curiosity_weight": reinforcement.curiosity_weight,
            "total_steps": reinforcement.total_steps,
            "total_reward": reinforcement.total_reward,
        })])

    # === 超参数 ===
    if hyper is not None:
        save_dict["__hyper__"] = np.array([json.dumps(hyper.values)])

    # === 情绪状态 ===
    if emotion is not None:
        emo_state = {
            "joy": float(emotion.state.joy),
            "sadness": float(emotion.state.sadness),
            "anger": float(emotion.state.anger),
            "fear": float(emotion.state.fear),
            "disgust": float(emotion.state.disgust),
            "surprise": float(emotion.state.surprise),
            "dopamine": float(emotion.dopamine.current_dopamine),
            "dopamine_baseline": float(emotion.dopamine.baseline),
            "dopamine_learning_rate": float(emotion.dopamine.learning_rate),
            "dopamine_discount": float(emotion.dopamine.discount),
            "decay_rate": float(emotion.decay_rate),
            "value_estimates": {k: float(v) for k, v in
                                emotion.dopamine.value_estimates.items()},
        }
        save_dict["__emotion__"] = np.array([json.dumps(emo_state, ensure_ascii=False)])

    # === 动机状态 ===
    if motivation is not None:
        mot_state = {
            "drives": {
                dt.value: {
                    "level": float(d.level),
                    "baseline": float(d.baseline),
                    "decay": float(d.decay),
                    "weight": float(d.weight),
                }
                for dt, d in motivation.drives.items()
            },
            "base_explore": float(getattr(motivation, 'base_explore', 0.2)),
        }
        save_dict["__motivation__"] = np.array([json.dumps(mot_state, ensure_ascii=False)])

    # === 意识状态 ===
    if consciousness is not None:
        con_state = {
            "noise_level": float(getattr(consciousness.workspace, 'noise_level', 0.1)),
            "broadcast_count": int(consciousness.metrics.broadcast_count),
            "phi": float(consciousness.metrics.phi),
        }
        save_dict["__consciousness__"] = np.array([json.dumps(con_state, ensure_ascii=False)])

    # === 记忆配置 ===
    if memory is not None:
        mem_state = {
            "sensory_capacity": memory.sensory_buffer.capacity,
            "stm_capacity": memory.short_term.capacity,
            "ltm_capacity": memory.long_term.capacity,
            "sensory_count": len(memory.sensory_buffer.buffer),
            "stm_count": len(memory.short_term.memory),
            "ltm_count": len(memory.long_term.memory),
        }
        save_dict["__memory__"] = np.array([json.dumps(mem_state, ensure_ascii=False)])

    # === 思考状态 ===
    if thought is not None:
        thought_state = {
            "space_capacity": thought.space.capacity,
            "vector_dim": int(thought.space.thoughts[0].content.shape[0])
                           if thought.space.thoughts else 512,
            "space_size": len(thought.space.thoughts),
            "stream_length": len(thought.stream.stream),
        }
        save_dict["__thought__"] = np.array([json.dumps(thought_state, ensure_ascii=False)])

    # === 注意力状态 ===
    if attention is not None:
        att_state = {
            "mode": attention.mode.name,
            "focus_source": attention.focus_source,
            "focus_strength": float(attention.focus_strength),
            "focus_duration": attention.focus_duration,
            "top_down_strength": float(attention.top_down_strength),
            "total_orienting": attention.total_orienting,
            "attention_switches": attention.attention_switches,
        }
        save_dict["__attention__"] = np.array([json.dumps(att_state, ensure_ascii=False)])

    # === 睡眠状态 ===
    if sleep is not None:
        slp_state = {
            "total_replays": sleep.total_replays,
            "total_consolidations": sleep.total_consolidations,
            "total_synaptic_scaling": sleep.total_synaptic_scaling,
            "replay_buffer_size": len(sleep.replay_buffer),
            "cycles_completed": sleep.cycles_completed,
        }
        save_dict["__sleep__"] = np.array([json.dumps(slp_state, ensure_ascii=False)])

    # === 稳态状态 ===
    if homeostasis is not None:
        hom_state = {
            "energy": float(homeostasis.energy),
            "fatigue": float(homeostasis.fatigue),
            "sleep_pressure": float(homeostasis.sleep_pressure),
            "hour": float(homeostasis.circadian.hour),
            "needs": {n.type.value: float(n.level) for n in homeostasis.needs.values()},
        }
        save_dict["__homeostasis__"] = np.array([json.dumps(hom_state, ensure_ascii=False)])

    # === 默认模式网络状态 ===
    if default_mode is not None:
        dmn_state = {
            "total_thoughts": default_mode.total_thoughts,
            "memory_fragments": len(default_mode.memory_fragments),
            "chain_counter": default_mode.chain_counter,
            "theme_counts": {t.name: c for t, c in default_mode.theme_counts.items()},
        }
        save_dict["__dmn__"] = np.array([json.dumps(dmn_state, ensure_ascii=False)])

    # === 元认知状态 ===
    if metacognition is not None:
        mc = metacognition
        meta_state = {
            "confidence": float(mc.confidence),
            "processing_fluency": float(mc.processing_fluency),
            "self_efficacy": float(mc.self_model.self_efficacy),
            "self_esteem": float(mc.self_model.self_esteem),
            "total_reflections": mc.self_model.total_reflections,
            "error_detections": mc.self_model.error_detections,
            "beliefs": {
                domain: {
                    "ability": float(b.ability),
                    "experience": b.experience,
                    "success": b.success,
                    "failure": b.failure,
                }
                for domain, b in mc.self_model.beliefs.items()
            },
        }
        save_dict["__metacognition__"] = np.array([json.dumps(meta_state, ensure_ascii=False)])

    # === 目标管理状态 ===
    if goals is not None:
        goals_state = {
            "total_completed": goals.total_completed,
            "total_abandoned": goals.total_abandoned,
            "total_interrupts": goals.total_interrupts,
            "active_goals": [
                {
                    "id": gid,
                    "description": goals.goals[gid].description,
                    "priority": goals.goals[gid].priority.label,
                    "progress": float(goals.goals[gid].progress),
                    "status": goals.goals[gid].status.value,
                }
                for gid in goals.goal_stack
            ],
        }
        save_dict["__goals__"] = np.array([json.dumps(goals_state, ensure_ascii=False)])

    # === 情绪调节状态 ===
    if emotion_regulation is not None:
        er = emotion_regulation
        er_state = {
            "emotional_stability": float(er.emotional_stability),
            "regulation_skill": float(er.regulation_skill),
            "wellbeing": float(er.wellbeing),
            "rumination_count": er.rumination_count,
            "rumination_tendency": float(er.rumination_tendency),
        }
        save_dict["__emotion_regulation__"] = np.array([json.dumps(er_state, ensure_ascii=False)])

    # 保存
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    np.savez_compressed(filepath, **save_dict)

    file_size = os.path.getsize(filepath)
    print(f"模型已导出: {filepath}")
    print(f"  版本: {meta['version']}")
    print(f"  模块数: {len(meta['modules'])}")
    print(f"  参数量: {meta['total_params']:,}")
    print(f"  文件大小: {file_size / 1024:.1f} KB")
    print(f"  时间: {meta['time_str']}")

    return filepath


def load_model(filepath: str) -> Tuple[PredictiveCodingNetwork, Dict[str, Any]]:
    """从 .npz 文件加载模型"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"模型文件不存在: {filepath}")

    data = np.load(filepath, allow_pickle=True)
    state = {}

    meta = json.loads(str(data["__meta__"][0]))
    state["meta"] = meta

    # 重建预测编码网络
    layer_sizes = meta.get("layer_sizes", DEFAULT_LAYER_SIZES)
    pc = PredictiveCodingNetwork(
        layer_sizes=layer_sizes,
        learning_rate=meta.get("pc_learning_rate", 0.01),
        time_constant=meta.get("pc_time_constant", 10.0),
    )

    # 恢复PC权重
    weight_info = json.loads(str(data["__pc_weight_shapes__"][0]))
    flat_weights = data["pc_weights"]
    offset = 0
    for name, shape in zip(weight_info["names"], weight_info["shapes"]):
        parts = name.split("_", 2)
        layer_idx = int(parts[1].replace("layer", ""))
        wname = parts[2]
        size = int(np.prod(shape))
        w = flat_weights[offset:offset + size].reshape(shape)
        setattr(pc.layers[layer_idx], wname, w)
        offset += size

    for i in range(len(pc.layers)):
        pc.layers[i].activation = data[f"pc_layer{i}_activation"].copy()
        pc.layers[i].prediction = data[f"pc_layer{i}_prediction"].copy()
        pc.layers[i].error = data[f"pc_layer{i}_error"].copy()
        pc.layers[i].precision = float(data[f"pc_layer{i}_precision"][0])

    # 辅助函数
    def load_json(key):
        if key in data:
            return json.loads(str(data[key][0]))
        return None

    state["hyper"] = HyperParams(load_json("__hyper__")) if "__hyper__" in data else None
    state["emotion"] = load_json("__emotion__")
    state["motivation"] = load_json("__motivation__")
    state["consciousness"] = load_json("__consciousness__")
    state["memory"] = load_json("__memory__")
    state["thought"] = load_json("__thought__")
    state["basal_ganglia"] = {
        "d1_weights": data["bg_d1_weights"] if "bg_d1_weights" in data else None,
        "d2_weights": data["bg_d2_weights"] if "bg_d2_weights" in data else None,
        "critic_weights": data["bg_critic_weights"] if "bg_critic_weights" in data else None,
        "actions": load_json("__bg_actions__"),
        "state": load_json("__bg_state__"),
    }
    state["cerebellum"] = {
        "granular_weights": data["cer_granular_weights"] if "cer_granular_weights" in data else None,
        "purkinje_weights": data["cer_purkinje_weights"] if "cer_purkinje_weights" in data else None,
        "timing_weights": data["cer_timing_weights"] if "cer_timing_weights" in data else None,
        "state": load_json("__cer_state__"),
    }
    state["reinforcement"] = {
        "critic_weights": data["rl_critic_weights"] if "rl_critic_weights" in data else None,
        "actor_weights": data["rl_actor_weights"] if "rl_actor_weights" in data else None,
        "state": load_json("__rl_state__"),
    }
    state["attention"] = load_json("__attention__")
    state["sleep"] = load_json("__sleep__")
    state["homeostasis"] = load_json("__homeostasis__")
    state["default_mode"] = load_json("__dmn__")
    state["metacognition"] = load_json("__metacognition__")
    state["goals"] = load_json("__goals__")
    state["emotion_regulation"] = load_json("__emotion_regulation__")

    print(f"模型已加载: {filepath}")
    print(f"  版本: {meta['version']}")
    print(f"  模块数: {len(meta.get('modules', []))}")
    print(f"  参数量: {meta['total_params']:,}")
    print(f"  层结构: {layer_sizes}")

    return pc, state


def restore_modules(state: Dict[str, Any],
                    emotion: EmotionalCore = None,
                    motivation: MotivationSystem = None,
                    consciousness: ConsciousnessSystem = None,
                    basal_ganglia=None,
                    cerebellum=None,
                    reinforcement=None,
                    metacognition=None):
    """将加载的状态恢复到各模块"""
    # 情绪
    if emotion and state.get("emotion"):
        es = state["emotion"]
        emotion.state.joy = es["joy"]
        emotion.state.sadness = es["sadness"]
        emotion.state.anger = es["anger"]
        emotion.state.fear = es["fear"]
        emotion.state.disgust = es["disgust"]
        emotion.state.surprise = es["surprise"]
        emotion.dopamine.current_dopamine = es["dopamine"]
        emotion.dopamine.baseline = es["dopamine_baseline"]
        emotion.dopamine.learning_rate = es["dopamine_learning_rate"]
        emotion.dopamine.discount = es["dopamine_discount"]
        emotion.decay_rate = es["decay_rate"]
        emotion.dopamine.value_estimates.update(es.get("value_estimates", {}))

    # 动机
    if motivation and state.get("motivation"):
        from .motivation import DriveType
        ms = state["motivation"]
        for drive_name, drive_data in ms["drives"].items():
            dt = DriveType(drive_name)
            if dt in motivation.drives:
                motivation.drives[dt].level = drive_data["level"]
                motivation.drives[dt].baseline = drive_data["baseline"]
                motivation.drives[dt].decay = drive_data["decay"]
                motivation.drives[dt].weight = drive_data["weight"]
        if hasattr(motivation, 'base_explore'):
            motivation.base_explore = ms.get("base_explore", 0.2)

    # 意识
    if consciousness and state.get("consciousness"):
        cs = state["consciousness"]
        if hasattr(consciousness.workspace, 'noise_level'):
            consciousness.workspace.noise_level = cs["noise_level"]
        consciousness.metrics.broadcast_count = cs["broadcast_count"]
        consciousness.metrics.phi = cs["phi"]

    # 基底神经节
    if basal_ganglia and state.get("basal_ganglia"):
        bg = state["basal_ganglia"]
        if bg["d1_weights"] is not None:
            basal_ganglia.d1_weights = bg["d1_weights"].copy()
            basal_ganglia.d2_weights = bg["d2_weights"].copy()
            basal_ganglia.critic_weights = bg["critic_weights"].copy()
        if bg["actions"]:
            from .basal_ganglia import Action, ActionType
            for aid, adata in bg["actions"].items():
                action = Action(
                    id=aid, name=adata["name"],
                    action_type=ActionType[adata["action_type"]],
                    q_value=adata["q_value"],
                    habit_strength=adata["habit_strength"],
                    execution_count=adata["execution_count"],
                )
                action.success_count = adata["success_count"]
                basal_ganglia.actions[aid] = action
            if bg["state"]:
                basal_ganglia.total_selections = bg["state"]["total_selections"]
                basal_ganglia.habit_selections = bg["state"]["habit_selections"]

    # 小脑
    if cerebellum and state.get("cerebellum"):
        cer = state["cerebellum"]
        if cer["granular_weights"] is not None:
            cerebellum.granular_weights = cer["granular_weights"].copy()
            cerebellum.purkinje_weights = cer["purkinje_weights"].copy()
            cerebellum.timing_weights = cer["timing_weights"].copy()

    # 强化学习
    if reinforcement and state.get("reinforcement"):
        rl = state["reinforcement"]
        if rl["critic_weights"] is not None:
            reinforcement.critic.weights = rl["critic_weights"].copy()
            reinforcement.actor.weights = rl["actor_weights"].copy()
        if rl["state"]:
            reinforcement.total_steps = rl["state"]["total_steps"]
            reinforcement.total_reward = rl["state"]["total_reward"]

    # 元认知
    if metacognition and state.get("metacognition"):
        mc = state["metacognition"]
        metacognition.confidence = mc["confidence"]
        metacognition.processing_fluency = mc["processing_fluency"]
        metacognition.self_model.self_efficacy = mc["self_efficacy"]
        metacognition.self_model.self_esteem = mc["self_esteem"]
        metacognition.self_model.total_reflections = mc["total_reflections"]
        metacognition.self_model.error_detections = mc["error_detections"]
        for domain, bdata in mc.get("beliefs", {}).items():
            if domain in metacognition.self_model.beliefs:
                b = metacognition.self_model.beliefs[domain]
                b.ability = bdata["ability"]
                b.experience = bdata["experience"]
                b.success = bdata["success"]
                b.failure = bdata["failure"]
