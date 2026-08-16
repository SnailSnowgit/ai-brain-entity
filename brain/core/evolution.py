"""
进化模块 — 遗传算法优化类脑系统参数

基因组成:
  - 预测编码网络权重 (998,402参数)
  - 超参数基因 (学习率/情绪衰减/多巴胺基线/好奇心等)

进化流程:
  选择(锦标赛) → 交叉(均匀) → 变异(高斯) → 精英保留
"""

import numpy as np
import copy
from typing import List, Dict, Any, Optional, Callable, Tuple


class HyperParams:
    """可进化的超参数集合"""

    # 超参数定义: (名称, 最小值, 最大值, 变异标准差)
    DEFINITIONS = [
        ("pc_learning_rate",    0.001, 0.05,  0.003),
        ("pc_precision_input",  0.5,   2.0,   0.15),
        ("pc_precision_mid",    0.3,   1.5,   0.12),
        ("pc_precision_top",    0.1,   1.0,   0.08),
        ("emotion_decay",       0.01,  0.15,  0.015),
        ("dopamine_baseline",   0.1,   0.6,   0.04),
        ("dopamine_learning",   0.05,  0.5,   0.04),
        ("curiosity_weight",    0.2,   1.5,   0.12),
        ("explore_base",        0.05,  0.4,   0.04),
        ("memory_consolidation",0.01,  0.2,   0.02),
        ("thought_decay",       0.005, 0.08,  0.008),
        ("conscious_noise",     0.02,  0.25,  0.025),
        ("valence_sensitivity", 0.3,   2.0,   0.15),
        ("arousal_sensitivity", 0.3,   2.0,   0.15),
        ("hebbian_rate",        0.005, 0.08,  0.008),
    ]

    def __init__(self, values: Dict[str, float] = None):
        if values:
            self.values = values
        else:
            self.values = {
                name: (lo + hi) / 2
                for name, lo, hi, _ in self.DEFINITIONS
            }

    def get(self, name: str) -> float:
        return self.values[name]

    def to_array(self) -> np.ndarray:
        return np.array([self.values[name] for name, _, _, _ in self.DEFINITIONS])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "HyperParams":
        values = {}
        for i, (name, lo, hi, _) in enumerate(cls.DEFINITIONS):
            values[name] = float(np.clip(arr[i], lo, hi))
        return cls(values)

    def mutate(self, rate: float = 0.2) -> "HyperParams":
        """高斯变异"""
        arr = self.to_array()
        for i, (name, lo, hi, sigma) in enumerate(self.DEFINITIONS):
            if np.random.random() < rate:
                arr[i] += np.random.normal(0, sigma)
                arr[i] = np.clip(arr[i], lo, hi)
        return HyperParams.from_array(arr)

    @classmethod
    def crossover(cls, p1: "HyperParams", p2: "HyperParams",
                  rate: float = 0.5) -> "HyperParams":
        """均匀交叉"""
        a1, a2 = p1.to_array(), p2.to_array()
        mask = np.random.random(len(a1)) < rate
        child = np.where(mask, a1, a2)
        return HyperParams.from_array(child)

    def __repr__(self):
        items = ", ".join(f"{k}={v:.3f}" for k, v in list(self.values.items())[:5])
        return f"HyperParams({items}, ...)"


class Genome:
    """
    个体基因组

    genes:
      - weights: 预测编码网络权重向量 (拼接)
      - hyper: HyperParams 超参数
    """

    def __init__(self, weights: np.ndarray, hyper: HyperParams,
                 fitness: float = -float("inf")):
        self.weights = weights
        self.hyper = hyper
        self.fitness = fitness

    def mutate(self, weight_rate: float = 0.05,
               weight_sigma: float = 0.02,
               hyper_rate: float = 0.2) -> "Genome":
        """变异: 权重高斯变异 + 超参数变异"""
        # 权重变异(稀疏)
        mask = np.random.random(len(self.weights)) < weight_rate
        new_weights = self.weights.copy()
        new_weights[mask] += np.random.normal(0, weight_sigma, mask.sum())
        np.clip(new_weights, -5, 5, out=new_weights)
        # 超参数变异
        new_hyper = self.hyper.mutate(rate=hyper_rate)
        return Genome(new_weights, new_hyper)

    @staticmethod
    def crossover(p1: "Genome", p2: "Genome",
                  weight_rate: float = 0.5) -> "Genome":
        """均匀交叉"""
        mask = np.random.random(len(p1.weights)) < weight_rate
        child_weights = np.where(mask, p1.weights, p2.weights)
        child_hyper = HyperParams.crossover(p1.hyper, p2.hyper)
        return Genome(child_weights, child_hyper)


class GeneticAlgorithm:
    """
    遗传算法

    用法:
        ga = GeneticAlgorithm(population_size=20, weight_dim=998402)
        for gen in range(50):
            for individual in ga.population:
                individual.fitness = evaluate(individual)
            ga.evolve()
    """

    def __init__(self,
                 population_size: int = 20,
                 weight_dim: int = 998402,
                 weight_init_scale: float = 0.1,
                 tournament_k: int = 3,
                 elite_count: int = 2,
                 crossover_rate: float = 0.7,
                 mutation_rate: float = 0.15,
                 weight_mut_rate: float = 0.03,
                 weight_mut_sigma: float = 0.015,
                 hyper_mut_rate: float = 0.25,
                 seed: int = None):
        """
        Args:
            population_size: 种群大小
            weight_dim: 权重维度(预测编码网络参数总数)
            weight_init_scale: 初始权重尺度
            tournament_k: 锦标赛选择k值
            elite_count: 精英保留数
            crossover_rate: 交叉概率
            mutation_rate: 个体变异概率
            weight_mut_rate: 权重变异比例
            weight_mut_sigma: 权重变异标准差
            hyper_mut_rate: 超参数变异概率
        """
        if seed is not None:
            np.random.seed(seed)

        self.pop_size = population_size
        self.weight_dim = weight_dim
        self.weight_init_scale = weight_init_scale
        self.tournament_k = tournament_k
        self.elite_count = elite_count
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.weight_mut_rate = weight_mut_rate
        self.weight_mut_sigma = weight_mut_sigma
        self.hyper_mut_rate = hyper_mut_rate

        self.generation = 0
        self.best_fitness_history: List[float] = []
        self.avg_fitness_history: List[float] = []
        self.best_genome: Optional[Genome] = None

        # 初始化种群
        self.population = self._init_population()

    def _init_population(self) -> List[Genome]:
        """初始化种群: 一个中心个体 + 随机变异"""
        population = []
        # 中心个体(He初始化尺度)
        center_weights = np.random.randn(self.weight_dim) * self.weight_init_scale
        center_hyper = HyperParams()
        population.append(Genome(center_weights, center_hyper))

        # 其余个体通过变异产生
        for _ in range(self.pop_size - 1):
            mut = population[0].mutate(
                weight_rate=0.3,
                weight_sigma=self.weight_init_scale * 2,
                hyper_rate=0.5,
            )
            population.append(mut)

        return population

    def _tournament_select(self) -> Genome:
        """锦标赛选择"""
        contenders = np.random.choice(
            self.pop_size, self.tournament_k, replace=False)
        best = None
        for idx in contenders:
            ind = self.population[idx]
            if best is None or ind.fitness > best.fitness:
                best = ind
        return best

    def evolve(self) -> Dict[str, Any]:
        """
        进化一代

        Returns:
            统计信息 dict
        """
        # 按适应度排序
        ranked = sorted(self.population,
                        key=lambda g: g.fitness, reverse=True)

        # 记录历史
        best_f = ranked[0].fitness
        avg_f = np.mean([g.fitness for g in self.population])
        self.best_fitness_history.append(best_f)
        self.avg_fitness_history.append(avg_f)

        if self.best_genome is None or best_f > self.best_genome.fitness:
            self.best_genome = copy.deepcopy(ranked[0])

        # 新一代
        new_pop = []

        # 精英保留
        for i in range(min(self.elite_count, len(ranked))):
            new_pop.append(copy.deepcopy(ranked[i]))

        # 产生后代
        while len(new_pop) < self.pop_size:
            parent1 = self._tournament_select()
            parent2 = self._tournament_select()

            if np.random.random() < self.crossover_rate:
                child = Genome.crossover(parent1, parent2)
            else:
                child = copy.deepcopy(parent1)

            if np.random.random() < self.mutation_rate:
                child = child.mutate(
                    weight_rate=self.weight_mut_rate,
                    weight_sigma=self.weight_mut_sigma,
                    hyper_rate=self.hyper_mut_rate,
                )

            new_pop.append(child)

        self.population = new_pop
        self.generation += 1

        return {
            "generation": self.generation,
            "best_fitness": best_f,
            "avg_fitness": avg_f,
            "worst_fitness": ranked[-1].fitness,
            "best_hyper": ranked[0].hyper,
        }

    def get_diversity(self) -> float:
        """计算种群多样性(权重平均成对距离)"""
        if self.pop_size < 2:
            return 0.0
        # 采样计算
        n_sample = min(10, self.pop_size)
        indices = np.random.choice(self.pop_size, n_sample, replace=False)
        dists = []
        for i in range(n_sample):
            for j in range(i + 1, n_sample):
                w1 = self.population[indices[i]].weights
                w2 = self.population[indices[j]].weights
                dists.append(np.mean((w1 - w2) ** 2))
        return float(np.mean(dists))

    def summary(self) -> str:
        if not self.best_fitness_history:
            return "尚未进化"
        return (
            f"代数={self.generation} "
            f"最佳={self.best_fitness_history[-1]:.4f} "
            f"平均={self.avg_fitness_history[-1]:.4f} "
            f"多样性={self.get_diversity():.6f}"
        )


def flatten_pc_weights(pc_network) -> np.ndarray:
    """将预测编码网络权重展平为向量"""
    parts = []
    for layer in pc_network.layers:
        if layer.top_down_weights.size > 1:
            parts.append(layer.top_down_weights.flatten())
            parts.append(layer.bottom_up_weights.flatten())
    return np.concatenate(parts)


def unflatten_pc_weights(pc_network, weights: np.ndarray):
    """将权重向量写回预测编码网络"""
    offset = 0
    for layer in pc_network.layers:
        if layer.top_down_weights.size > 1:
            td_size = layer.top_down_weights.size
            bu_size = layer.bottom_up_weights.size
            layer.top_down_weights = weights[offset:offset + td_size].reshape(
                layer.top_down_weights.shape)
            offset += td_size
            layer.bottom_up_weights = weights[offset:offset + bu_size].reshape(
                layer.bottom_up_weights.shape)
            offset += bu_size


def apply_hyperparams(brain, hyper: HyperParams):
    """将超参数应用到类脑系统"""
    h = hyper.values
    # 预测编码
    brain.predictor.lr = h["pc_learning_rate"]
    brain.predictor.layers[0].precision = h["pc_precision_input"]
    brain.predictor.layers[1].precision = h["pc_precision_mid"]
    brain.predictor.layers[2].precision = h["pc_precision_top"]
    # 情绪
    brain.emotion.decay_rate = h["emotion_decay"]
    brain.emotion.dopamine.baseline = h["dopamine_baseline"]
    brain.emotion.dopamine.learning_rate = h["dopamine_learning"]
    # 动机
    brain.motivation.base_explore = h["explore_base"]
    # 意识
    brain.consciousness.workspace.noise_level = h["conscious_noise"]
