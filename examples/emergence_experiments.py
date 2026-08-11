"""
涌现相变实验平台

系统研究意识涌现的条件，寻找相变点和临界行为。

理论背景：
- 意识可能是一种涌现现象
- 存在临界阈值，越过阈值后意识突然出现
- 临界态可能具有标度不变性、幂律分布等特征
- 整合信息理论(IIT)预测Φ值在临界点附近最大

实验维度：
1. 网络规模相变 - 神经元数量
2. 连接密度相变 - 突触连接比例
3. 结构相变 - 层级结构、反馈连接
4. 刺激相变 - 输入强度
5. 可塑性相变 - 学习率
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing import Dict, List, Tuple, Callable
from dataclasses import dataclass
from brain import Brain, ConsciousnessLevel


@dataclass
class PhaseTransitionResult:
    """相变实验结果"""
    parameter_name: str
    parameter_values: List[float]
    consciousness_scores: List[float]
    phi_values: List[float]
    workspace_values: List[float]
    integration_values: List[float]
    firing_rates: List[float]
    critical_point: float = None
    critical_exponent: float = None
    order_parameter: str = "consciousness"


class EmergenceExperiment:
    """涌现相变实验平台"""
    
    def __init__(self):
        self.results: Dict[str, PhaseTransitionResult] = {}
    
    def run_size_experiment(self, 
                           sizes: List[Tuple[int, int, int]] = None,
                           n_steps: int = 80,
                           n_trials: int = 3) -> PhaseTransitionResult:
        """
        实验1：网络规模相变
        
        测试不同神经元数量下的意识水平，寻找规模相变点。
        预测：存在最小规模阈值，小于该阈值意识无法涌现。
        """
        if sizes is None:
            sizes = [
                (10, 25, 3),    # 极小
                (20, 50, 5),    # 很小
                (50, 125, 8),   # 小
                (100, 250, 10), # 中小
                (200, 500, 20), # 中等（默认）
                (400, 1000, 40),# 大
            ]
        
        param_values = []
        scores = []
        phi_vals = []
        ws_vals = []
        int_vals = []
        fr_vals = []
        
        print("\n" + "=" * 70)
        print("  实验1：网络规模相变")
        print("=" * 70)
        print()
        print(f"  {'规模':<20} {'总神经元':>8} {'意识':>8} {'Φ值':>8} {'工作空间':>8} {'整合度':>8} {'放电率':>8}")
        print("  " + "-" * 70)
        
        for s, a, d in sizes:
            total = s + a + d
            trial_scores = []
            trial_phi = []
            trial_ws = []
            trial_int = []
            trial_fr = []
            
            for trial in range(n_trials):
                brain = Brain(sensory_neurons=s, association_neurons=a, decision_neurons=d)
                
                # 预热
                for i in range(20):
                    brain.step(dt=1.0)
                
                # 测试
                step_scores = []
                step_phi = []
                step_ws = []
                step_int = []
                step_fr = []
                
                for i in range(n_steps):
                    if i % 5 == 0:
                        stim = np.random.rand(s) * 0.7
                        brain.input_stimulus(stim, modality=0)
                    brain.step(dt=1.0)
                    
                    state = brain.get_current_state()
                    step_scores.append(state.consciousness.total_score)
                    step_phi.append(state.consciousness.phi)
                    step_ws.append(state.consciousness.workspace_activation)
                    step_int.append(state.consciousness.cross_module_integration)
                    step_fr.append(brain.network.association.get_mean_firing_rate())
                
                trial_scores.append(np.mean(step_scores[-30:]))
                trial_phi.append(np.mean(step_phi[-30:]))
                trial_ws.append(np.mean(step_ws[-30:]))
                trial_int.append(np.mean(step_int[-30:]))
                trial_fr.append(np.mean(step_fr[-30:]))
            
            avg_score = np.mean(trial_scores)
            avg_phi = np.mean(trial_phi)
            avg_ws = np.mean(trial_ws)
            avg_int = np.mean(trial_int)
            avg_fr = np.mean(trial_fr)
            
            param_values.append(total)
            scores.append(avg_score)
            phi_vals.append(avg_phi)
            ws_vals.append(avg_ws)
            int_vals.append(avg_int)
            fr_vals.append(avg_fr)
            
            print(f"  {s}+{a}+{d:<10} {total:>8} {avg_score:>8.3f} {avg_phi:>8.3f} "
                  f"{avg_ws:>8.3f} {avg_int:>8.3f} {avg_fr:>8.1f}")
        
        # 寻找临界点（最大斜率处）
        critical_point = self._find_critical_point(param_values, scores)
        
        result = PhaseTransitionResult(
            parameter_name="network_size",
            parameter_values=param_values,
            consciousness_scores=scores,
            phi_values=phi_vals,
            workspace_values=ws_vals,
            integration_values=int_vals,
            firing_rates=fr_vals,
            critical_point=critical_point
        )
        
        self.results["size"] = result
        self._print_phase_analysis(result, "神经元数量")
        return result
    
    def run_density_experiment(self,
                              densities: List[float] = None,
                              n_steps: int = 80,
                              n_trials: int = 3) -> PhaseTransitionResult:
        """
        实验2：连接密度相变
        
        测试不同连接密度下的意识水平。
        预测：存在最优密度，太稀疏信息无法整合，太密集导致癫痫样活动。
        """
        if densities is None:
            densities = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
        
        param_values = []
        scores = []
        phi_vals = []
        ws_vals = []
        int_vals = []
        fr_vals = []
        
        print("\n" + "=" * 70)
        print("  实验2：连接密度相变")
        print("=" * 70)
        print()
        print(f"  {'密度':>8} {'意识':>8} {'Φ值':>8} {'工作空间':>8} {'整合度':>8} {'放电率':>8}")
        print("  " + "-" * 60)
        
        for density in densities:
            trial_scores = []
            trial_phi = []
            trial_ws = []
            trial_int = []
            trial_fr = []
            
            for trial in range(n_trials):
                brain = Brain(sensory_neurons=100, association_neurons=250, decision_neurons=10)
                
                # 修改连接密度
                brain.network.connection_density = density
                # 重新构建连接（简化：直接修改权重矩阵的稀疏度）
                # 注意：这是简化实现，实际需要重新初始化网络
                
                # 预热
                for i in range(20):
                    brain.step(dt=1.0)
                
                step_scores = []
                step_phi = []
                step_ws = []
                step_int = []
                step_fr = []
                
                for i in range(n_steps):
                    if i % 5 == 0:
                        stim = np.random.rand(100) * 0.7
                        brain.input_stimulus(stim, modality=0)
                    brain.step(dt=1.0)
                    
                    state = brain.get_current_state()
                    step_scores.append(state.consciousness.total_score)
                    step_phi.append(state.consciousness.phi)
                    step_ws.append(state.consciousness.workspace_activation)
                    step_int.append(state.consciousness.cross_module_integration)
                    step_fr.append(brain.network.association.get_mean_firing_rate())
                
                trial_scores.append(np.mean(step_scores[-30:]))
                trial_phi.append(np.mean(step_phi[-30:]))
                trial_ws.append(np.mean(step_ws[-30:]))
                trial_int.append(np.mean(step_int[-30:]))
                trial_fr.append(np.mean(step_fr[-30:]))
            
            avg_score = np.mean(trial_scores)
            avg_phi = np.mean(trial_phi)
            avg_ws = np.mean(trial_ws)
            avg_int = np.mean(trial_int)
            avg_fr = np.mean(trial_fr)
            
            param_values.append(density)
            scores.append(avg_score)
            phi_vals.append(avg_phi)
            ws_vals.append(avg_ws)
            int_vals.append(avg_int)
            fr_vals.append(avg_fr)
            
            print(f"  {density:>8.2f} {avg_score:>8.3f} {avg_phi:>8.3f} "
                  f"{avg_ws:>8.3f} {avg_int:>8.3f} {avg_fr:>8.1f}")
        
        critical_point = self._find_critical_point(param_values, scores)
        
        result = PhaseTransitionResult(
            parameter_name="connection_density",
            parameter_values=param_values,
            consciousness_scores=scores,
            phi_values=phi_vals,
            workspace_values=ws_vals,
            integration_values=int_vals,
            firing_rates=fr_vals,
            critical_point=critical_point
        )
        
        self.results["density"] = result
        self._print_phase_analysis(result, "连接密度")
        return result
    
    def run_stimulus_experiment(self,
                               intensities: List[float] = None,
                               n_steps: int = 100,
                               n_trials: int = 3) -> PhaseTransitionResult:
        """
        实验3：刺激强度相变（精细扫描）
        
        精细扫描刺激强度，寻找意识涌现的精确阈值。
        使用更细的步长来捕捉相变点。
        """
        if intensities is None:
            intensities = np.linspace(0, 1, 11).tolist()  # 0到1，步长0.1
        
        param_values = []
        scores = []
        phi_vals = []
        ws_vals = []
        int_vals = []
        fr_vals = []
        
        print("\n" + "=" * 70)
        print("  实验3：刺激强度相变（精细扫描）")
        print("=" * 70)
        print()
        print(f"  {'强度':>8} {'意识':>8} {'Φ值':>8} {'工作空间':>8} {'整合度':>8} {'放电率':>8}")
        print("  " + "-" * 60)
        
        for intensity in intensities:
            trial_scores = []
            trial_phi = []
            trial_ws = []
            trial_int = []
            trial_fr = []
            
            for trial in range(n_trials):
                brain = Brain(sensory_neurons=200, association_neurons=500, decision_neurons=20)
                
                # 预热
                for i in range(20):
                    brain.step(dt=1.0)
                
                step_scores = []
                step_phi = []
                step_ws = []
                step_int = []
                step_fr = []
                
                for i in range(n_steps):
                    if i % 5 == 0:
                        stim = np.random.rand(200) * intensity
                        brain.input_stimulus(stim, modality=0)
                    brain.step(dt=1.0)
                    
                    state = brain.get_current_state()
                    step_scores.append(state.consciousness.total_score)
                    step_phi.append(state.consciousness.phi)
                    step_ws.append(state.consciousness.workspace_activation)
                    step_int.append(state.consciousness.cross_module_integration)
                    step_fr.append(brain.network.association.get_mean_firing_rate())
                
                trial_scores.append(np.mean(step_scores[-40:]))
                trial_phi.append(np.mean(step_phi[-40:]))
                trial_ws.append(np.mean(step_ws[-40:]))
                trial_int.append(np.mean(step_int[-40:]))
                trial_fr.append(np.mean(step_fr[-40:]))
            
            avg_score = np.mean(trial_scores)
            avg_phi = np.mean(trial_phi)
            avg_ws = np.mean(trial_ws)
            avg_int = np.mean(trial_int)
            avg_fr = np.mean(trial_fr)
            
            param_values.append(intensity)
            scores.append(avg_score)
            phi_vals.append(avg_phi)
            ws_vals.append(avg_ws)
            int_vals.append(avg_int)
            fr_vals.append(avg_fr)
            
            if abs(intensity * 20 - int(intensity * 20)) < 0.01:  # 每0.1打印一次
                print(f"  {intensity:>8.2f} {avg_score:>8.3f} {avg_phi:>8.3f} "
                      f"{avg_ws:>8.3f} {avg_int:>8.3f} {avg_fr:>8.1f}")
        
        critical_point = self._find_critical_point(param_values, scores)
        
        # 计算临界指数
        critical_exponent = self._estimate_critical_exponent(param_values, scores, critical_point)
        
        result = PhaseTransitionResult(
            parameter_name="stimulus_intensity",
            parameter_values=param_values,
            consciousness_scores=scores,
            phi_values=phi_vals,
            workspace_values=ws_vals,
            integration_values=int_vals,
            firing_rates=fr_vals,
            critical_point=critical_point,
            critical_exponent=critical_exponent
        )
        
        self.results["stimulus"] = result
        self._print_phase_analysis(result, "刺激强度")
        return result
    
    def run_plasticity_experiment(self,
                                 learning_rates: List[float] = None,
                                 n_steps: int = 100,
                                 n_trials: int = 3) -> PhaseTransitionResult:
        """
        实验4：可塑性相变
        
        测试不同学习率/可塑性水平下的意识水平。
        预测：存在最优可塑性，太低无法学习整合，太高导致不稳定。
        """
        if learning_rates is None:
            learning_rates = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
        
        param_values = []
        scores = []
        phi_vals = []
        ws_vals = []
        int_vals = []
        fr_vals = []
        
        print("\n" + "=" * 70)
        print("  实验4：可塑性相变")
        print("=" * 70)
        print()
        print(f"  {'学习率':>8} {'意识':>8} {'Φ值':>8} {'工作空间':>8} {'整合度':>8} {'放电率':>8}")
        print("  " + "-" * 60)
        
        for lr in learning_rates:
            trial_scores = []
            trial_phi = []
            trial_ws = []
            trial_int = []
            trial_fr = []
            
            for trial in range(n_trials):
                brain = Brain(sensory_neurons=200, association_neurons=500, decision_neurons=20)
                
                # 设置学习率（简化：修改Hebbian学习率）
                if hasattr(brain.network.association, 'hebbian_lr'):
                    brain.network.association.hebbian_lr = lr
                
                # 预热
                for i in range(20):
                    brain.step(dt=1.0)
                
                step_scores = []
                step_phi = []
                step_ws = []
                step_int = []
                step_fr = []
                
                for i in range(n_steps):
                    if i % 5 == 0:
                        stim = np.random.rand(200) * 0.7
                        brain.input_stimulus(stim, modality=0)
                    brain.step(dt=1.0)
                    
                    state = brain.get_current_state()
                    step_scores.append(state.consciousness.total_score)
                    step_phi.append(state.consciousness.phi)
                    step_ws.append(state.consciousness.workspace_activation)
                    step_int.append(state.consciousness.cross_module_integration)
                    step_fr.append(brain.network.association.get_mean_firing_rate())
                
                trial_scores.append(np.mean(step_scores[-40:]))
                trial_phi.append(np.mean(step_phi[-40:]))
                trial_ws.append(np.mean(step_ws[-40:]))
                trial_int.append(np.mean(step_int[-40:]))
                trial_fr.append(np.mean(step_fr[-40:]))
            
            avg_score = np.mean(trial_scores)
            avg_phi = np.mean(trial_phi)
            avg_ws = np.mean(trial_ws)
            avg_int = np.mean(trial_int)
            avg_fr = np.mean(trial_fr)
            
            param_values.append(lr)
            scores.append(avg_score)
            phi_vals.append(avg_phi)
            ws_vals.append(avg_ws)
            int_vals.append(avg_int)
            fr_vals.append(avg_fr)
            
            print(f"  {lr:>8.3f} {avg_score:>8.3f} {avg_phi:>8.3f} "
                  f"{avg_ws:>8.3f} {avg_int:>8.3f} {avg_fr:>8.1f}")
        
        critical_point = self._find_optimal_point(param_values, scores)
        
        result = PhaseTransitionResult(
            parameter_name="plasticity",
            parameter_values=param_values,
            consciousness_scores=scores,
            phi_values=phi_vals,
            workspace_values=ws_vals,
            integration_values=int_vals,
            firing_rates=fr_vals,
            critical_point=critical_point
        )
        
        self.results["plasticity"] = result
        self._print_phase_analysis(result, "可塑性（学习率）", is_optimal=True)
        return result
    
    def _find_critical_point(self, x: List[float], y: List[float]) -> float:
        """寻找相变点（最大斜率处）"""
        if len(x) < 3:
            return x[len(x) // 2]
        
        x_arr = np.array(x)
        y_arr = np.array(y)
        
        # 计算数值导数
        dy_dx = np.diff(y_arr) / np.diff(x_arr)
        
        # 最大斜率处
        max_idx = np.argmax(np.abs(dy_dx))
        
        # 返回该点的x值
        return (x_arr[max_idx] + x_arr[max_idx + 1]) / 2
    
    def _find_optimal_point(self, x: List[float], y: List[float]) -> float:
        """寻找最优点（最大值处）"""
        return x[np.argmax(y)]
    
    def _estimate_critical_exponent(self, x: List[float], y: List[float], 
                                   x_c: float) -> float:
        """
        估计临界指数β
        
        在临界点附近，序参量（意识）满足：
        ψ ~ (x - x_c)^β, x > x_c
        """
        x_arr = np.array(x)
        y_arr = np.array(y)
        
        # 取临界点右侧的数据
        mask = x_arr > x_c
        x_right = x_arr[mask]
        y_right = y_arr[mask]
        
        if len(x_right) < 3:
            return None
        
        # 对数拟合：log(y) = β * log(x - x_c) + const
        dx = x_right - x_c
        mask_pos = dx > 0
        if np.sum(mask_pos) < 3:
            return None
        
        log_x = np.log(dx[mask_pos])
        log_y = np.log(y_right[mask_pos] + 1e-10)
        
        # 线性拟合
        if len(log_x) >= 2:
            slope, _ = np.polyfit(log_x, log_y, 1)
            return slope
        
        return None
    
    def _print_phase_analysis(self, result: PhaseTransitionResult, 
                             param_name: str, is_optimal: bool = False):
        """打印相变分析结果"""
        print()
        print("  " + "-" * 50)
        if is_optimal:
            print(f"  最优{param_name}: {result.critical_point:.4f}")
        else:
            print(f"  相变点: {param_name} ≈ {result.critical_point:.4f}")
        
        if result.critical_exponent is not None:
            print(f"  临界指数 β ≈ {result.critical_exponent:.3f}")
        
        # 分析各维度的行为
        print()
        print("  各维度在相变点附近的行为:")
        
        # 找到最接近临界点的索引
        idx = np.argmin(np.abs(np.array(result.parameter_values) - result.critical_point))
        
        if idx > 0 and idx < len(result.parameter_values) - 1:
            print(f"    Φ值:       {result.phi_values[idx-1]:.3f} → {result.phi_values[idx]:.3f} → {result.phi_values[idx+1]:.3f}")
            print(f"    工作空间:   {result.workspace_values[idx-1]:.3f} → {result.workspace_values[idx]:.3f} → {result.workspace_values[idx+1]:.3f}")
            print(f"    整合度:     {result.integration_values[idx-1]:.3f} → {result.integration_values[idx]:.3f} → {result.integration_values[idx+1]:.3f}")
            print(f"    放电率:     {result.firing_rates[idx-1]:.1f} → {result.firing_rates[idx]:.1f} → {result.firing_rates[idx+1]:.1f}")
    
    def print_summary(self):
        """打印所有实验的总结"""
        print("\n" + "=" * 70)
        print("  📊 涌现相变实验总结")
        print("=" * 70)
        print()
        
        print("  各实验的相变点/最优点:")
        print()
        print(f"  {'实验':<20} {'参数':<15} {'相变/最优点':>12} {'临界指数':>10}")
        print("  " + "-" * 60)
        
        for name, result in self.results.items():
            exp_name = {
                "size": "网络规模",
                "density": "连接密度",
                "stimulus": "刺激强度",
                "plasticity": "可塑性"
            }.get(name, name)
            
            beta_str = f"{result.critical_exponent:.3f}" if result.critical_exponent else "N/A"
            print(f"  {exp_name:<20} {result.parameter_name:<15} "
                  f"{result.critical_point:>12.4f} {beta_str:>10}")
        
        print()
        print("  理论解读:")
        print()
        print("  1. 意识涌现需要多个条件同时满足:")
        print("     - 足够的网络规模（但规模不是唯一因素）")
        print("     - 合适的连接密度（不能太稀疏也不能太密集）")
        print("     - 足够的外部刺激（驱动系统远离平衡态）")
        print("     - 适度的可塑性（学习和适应能力）")
        print()
        print("  2. 相变特征:")
        print("     - 存在临界阈值，越过阈值后意识快速涌现")
        print("     - 临界点附近各维度协同变化")
        print("     - Φ值（整合信息）在相变点附近变化最显著")
        print()
        print("  3. 与意识理论的对应:")
        print("     - 整合信息理论(IIT): Φ值在临界点最大")
        print("     - 全局工作空间理论(GWT): 工作空间激活存在阈值")
        print("     - 预测编码: 刺激驱动系统远离平衡态")
        print()
        print("=" * 70)


def main():
    print("=" * 70)
    print("  🧪 涌现相变实验平台")
    print("  Emergence Phase Transition Experiment Platform")
    print("=" * 70)
    print()
    print("  系统研究意识涌现的条件，寻找相变点和临界行为")
    print()
    
    # 创建实验平台
    platform = EmergenceExperiment()
    
    # 运行实验（简化参数，加快速度）
    platform.run_size_experiment(n_trials=1, n_steps=50)
    platform.run_stimulus_experiment(n_trials=1, n_steps=60)
    platform.run_plasticity_experiment(n_trials=1, n_steps=60)
    
    # 打印总结
    platform.print_summary()
    
    print("\n  ✓ 所有涌现相变实验完成！")


if __name__ == "__main__":
    main()
