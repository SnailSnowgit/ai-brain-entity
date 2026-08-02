# -*- coding: utf-8 -*-
"""果蝇蘑菇体冲突决策 SNN —— BrainCog 范式的零依赖复刻（v5.0 实验）

忠实移植自 BrainCog-X/Brain-Cog:
    examples/Brain_Cognitive_Function_Simulation/drosophila/drosophila.py
    braincog/model_zoo/linearNet.py    (droDMTrainNet)
    braincog/model_zoo/nonlinearNet.py (droDMTestNet)
神经科学出处: Zhao et al., Scientific Reports (2020)，果蝇蘑菇体
（复眼 -> KC -> MBON）在颜色/形状线索冲突下的趋避决策。

原版依赖 torch 自动微分实现 STDP（dw = pre_trace ⊗ post_spike）；
本文件用纯 Python 标准库展开同样的数学，结果与原实现对齐。

任务设定（复眼 5 维输入向量 [R, G, B, T, t]）：
    GT = [0, 0.8, 0, 1.0, 0]  绿色正立 T —— 安全
    Bt = [0, 0, 0.8, 0, 1.0]  蓝色倒立 T —— 危险（伴随惩罚）
训练: visual-KC 用普通 STDP; KC-MBON 用奖励调制迹 (R-STDP,
    trace_decay=0.8, 被惩罚动作列乘以 -1)。
测试: 冲突刺激 = 绿色 (安全色) + 倒立 T (危险形状)，颜色浓度 c 从 0
    扫到 1.1 共 12 点，每点 500 步计数两个 MBON 动作的脉冲数，
    PI = (t1 - t2) / (t1 + t2)。
两个版本:
    linear     —— 训练网络直接推理 (input -> visual -> KC -> MBON)
    non-linear —— 测试网络加入 APL 反馈抑制与 DA 多巴胺通路，
                 前 10 步隔步注入 0.5 的 DA 电流，决策出现非线性转折。

运行:
    python drosophila_decision.py            # 完整训练 + 两条 PI 曲线
    python drosophila_decision.py --plot     # 额外保存 figures/drosophila_pi.png
"""
import copy
import sys

# ---------------------------------------------------------------- 常量
NUM_STATE = 5
NUM_ACTION = 2
NUM_APL = 2
NUM_DA = 1
WEIGHT_EXC = 0.5
WEIGHT_INH = -0.05        # 训练网 MBON 互相抑制
WEIGHT_INH_TEST = -0.3    # 测试网 APL/DA 抑制
TRACE_DECAY = 0.8         # 奖励调制资格迹衰减
STDP_DECAY = 0.99         # STDP 预突触迹衰减
THRESHOLD = 0.5           # IFNode 阈值
TRAIN_ROUNDS = 20
TEST_NUM = 12             # 颜色浓度扫描点数 (c = 0, 0.1, ..., 1.1)
TEST_STEPS = 500


# ---------------------------------------------------------------- 小工具
def eye(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def ones(r, c):
    return [[1.0] * c for _ in range(r)]


def scale(w, s):
    return [[v * s for v in row] for row in w]


def x_dot_W(x, w):
    """行向量 x (in,) 乘矩阵 w (in, out) -> (out,)"""
    out_dim = len(w[0])
    out = [0.0] * out_dim
    for k, xk in enumerate(x):
        if xk == 0.0:
            continue
        row = w[k]
        for j in range(out_dim):
            out[j] += xk * row[j]
    return out


def v_add(a, b):
    return [x + y for x, y in zip(a, b)]


def relu(x):
    return [v if v > 0.0 else 0.0 for v in x]


def normalize_l1(w, dim):
    """F.normalize(p=1): dim=1 每行 L1 归一; dim=0 每列 L1 归一（零行/列跳过）"""
    if dim == 1:
        for row in w:
            s = sum(abs(v) for v in row)
            if s > 0.0:
                for j in range(len(row)):
                    row[j] /= s
    else:
        for j in range(len(w[0])):
            s = sum(abs(w[k][j]) for k in range(len(w)))
            if s > 0.0:
                for k in range(len(w)):
                    w[k][j] /= s


class IFNode:
    """Integrate-and-Fire：mem += input; mem > threshold -> spike 且硬复位到 0"""

    def __init__(self, n, threshold=THRESHOLD):
        self.mem = [0.0] * n
        self.threshold = threshold

    def __call__(self, inputs):
        spikes = []
        for j, i in enumerate(inputs):
            self.mem[j] += i
            s = 1.0 if self.mem[j] > self.threshold else 0.0
            self.mem[j] *= 1.0 - s
            spikes.append(s)
        return spikes

    def reset(self):
        self.mem = [0.0] * len(self.mem)


class STDPRule:
    """单/多输入 STDP: dw^m[k][j] = trace^m[k] * post_spike[j]"""

    def __init__(self, dims, decay=STDP_DECAY):
        self.decay = decay
        self.traces = [None] * len(dims)
        self.dims = dims

    def trace_update(self, idx, x):
        if self.traces[idx] is None:
            self.traces[idx] = list(x)
        else:
            t = self.traces[idx]
            for k in range(len(t)):
                t[k] = t[k] * self.decay + x[k]
        return self.traces[idx]

    def dw(self, trace, spikes):
        return [[t * s for s in spikes] for t in trace]

    def reset(self):
        self.traces = [None] * len(self.dims)


def update_weight(conn, dw, dim=1):
    """W += dw * mask; 然后按 dim 做 L1 归一"""
    w, mask = conn["w"], conn["mask"]
    for k in range(len(w)):
        for j in range(len(w[0])):
            w[k][j] += dw[k][j] * mask[k][j]
    normalize_l1(w, dim)


# ---------------------------------------------------------------- 训练网
class DroDMTrainNet:
    """input -> visual -> KC -> MBON，MBON 间互相抑制（线性版本）"""

    def __init__(self, conns):
        self.conns = conns
        self.node_vis = IFNode(NUM_STATE)
        self.node_kc = IFNode(NUM_STATE)
        self.node_mbon = IFNode(NUM_ACTION)
        self.rule_kc = STDPRule([NUM_STATE])
        self.rule_mbon = STDPRule([NUM_STATE, NUM_ACTION])
        self.out_mbon = [0.0] * NUM_ACTION

    def forward(self, x):
        out_vis = self.node_vis(x_dot_W(x, self.conns[0]["w"]))
        # vis -> kc, STDP
        trace_vis = self.rule_kc.trace_update(0, out_vis)
        out_kc = self.node_kc(x_dot_W(out_vis, self.conns[1]["w"]))
        dw_kc = self.rule_kc.dw(trace_vis, out_kc)
        # kc -> mbon (+ 上一步 mbon 互抑)
        trace_kc = self.rule_mbon.trace_update(0, out_kc)
        self.rule_mbon.trace_update(1, self.out_mbon)
        i_mbon = v_add(x_dot_W(out_kc, self.conns[2]["w"]),
                       x_dot_W(self.out_mbon, self.conns[3]["w"]))
        out_mbon = self.node_mbon(i_mbon)
        dw_mbon = self.rule_mbon.dw(trace_kc, out_mbon)
        self.out_mbon = out_mbon
        return out_mbon, dw_kc, dw_mbon

    def update(self, idx, dw):
        update_weight(self.conns[idx], dw, dim=1)

    def reset(self):
        self.node_vis.reset()
        self.node_kc.reset()
        self.node_mbon.reset()
        self.rule_kc.reset()
        self.rule_mbon.reset()
        self.out_mbon = [0.0] * NUM_ACTION


# ---------------------------------------------------------------- 测试网
class DroDMTestNet:
    """训练网 + APL 反馈抑制 + DA 多巴胺通路（非线性版本）"""

    def __init__(self, conns):
        self.conns = conns
        self.node_vis = IFNode(NUM_STATE)
        self.node_kc = IFNode(NUM_STATE)
        self.node_mbon = IFNode(NUM_ACTION)
        self.node_apl = IFNode(NUM_APL)
        self.node_da = IFNode(NUM_DA)
        self.rule_kc = STDPRule([NUM_STATE, NUM_APL])
        self.rule_mbon = STDPRule([NUM_STATE, NUM_ACTION, NUM_DA])
        self.rule_apl = STDPRule([NUM_STATE, NUM_DA])
        self.rule_da = STDPRule([NUM_APL, NUM_DA])
        self.out_kc = [0.0] * NUM_STATE
        self.out_mbon = [0.0] * NUM_ACTION
        self.out_apl = [0.0] * NUM_APL
        self.out_da = [0.0] * NUM_DA

    def forward(self, x, input_da):
        out_vis = self.node_vis(x_dot_W(x, self.conns[0]["w"]))
        # KC: vis + APL(上一步) 反馈抑制
        self.rule_kc.trace_update(0, out_vis)
        trace_apl = self.rule_kc.trace_update(1, self.out_apl)
        i_kc = v_add(x_dot_W(out_vis, self.conns[1]["w"]),
                     x_dot_W(self.out_apl, self.conns[5]["w"]))
        out_kc = self.node_kc(i_kc)
        dw_apl_kc = self.rule_kc.dw(trace_apl, out_kc)  # -> con5
        self.out_kc = out_kc
        # MBON: KC + MBON(上一步) 互抑 + DA(上一步)
        self.rule_mbon.trace_update(0, out_kc)
        self.rule_mbon.trace_update(1, self.out_mbon)
        self.rule_mbon.trace_update(2, self.out_da)
        i_mbon = v_add(x_dot_W(out_kc, self.conns[2]["w"]),
                       v_add(x_dot_W(self.out_mbon, self.conns[3]["w"]),
                             x_dot_W(self.out_da, self.conns[9]["w"])))
        out_mbon = self.node_mbon(i_mbon)
        self.out_mbon = out_mbon
        # APL: KC + DA(上一步)
        trace_kc = self.rule_apl.trace_update(0, out_kc)
        self.rule_apl.trace_update(1, self.out_da)
        i_apl = v_add(x_dot_W(out_kc, self.conns[4]["w"]),
                      x_dot_W(self.out_da, self.conns[6]["w"]))
        out_apl = self.node_apl(i_apl)
        dw_kc_apl = self.rule_apl.dw(trace_kc, out_apl)  # -> con4
        self.out_apl = out_apl
        # DA: APL(本步) + 外部 DA 输入
        self.rule_da.trace_update(0, out_apl)
        self.rule_da.trace_update(1, input_da)
        i_da = v_add(x_dot_W(out_apl, self.conns[7]["w"]),
                     x_dot_W(input_da, self.conns[8]["w"]))
        self.out_da = self.node_da(i_da)
        return out_mbon, dw_apl_kc, dw_kc_apl


# ---------------------------------------------------------------- 连接构建
def conn(w, mask=None):
    return {"w": w, "mask": mask if mask is not None else ones(len(w), len(w[0]))}


def build_train_conns():
    con0 = scale(eye(NUM_STATE), WEIGHT_EXC)                       # input-visual
    con1 = scale(eye(NUM_STATE), WEIGHT_EXC)                       # visual-kc
    con2 = scale(ones(NUM_STATE, NUM_ACTION), WEIGHT_EXC)          # kc-mbon
    off_diag = [[0.0 if i == j else 1.0 for j in range(NUM_ACTION)]
                for i in range(NUM_ACTION)]
    con3 = scale(off_diag, WEIGHT_INH)                             # mbon-mbon 互抑
    return [conn(con0, eye(NUM_STATE)), conn(con1, eye(NUM_STATE)),
            conn(con2, ones(NUM_STATE, NUM_ACTION)), conn(con3, off_diag)]


def build_test_conns(train_conns):
    """训练好的 0-3 号连接 + APL/DA 通路 4-9"""
    conns = copy.deepcopy(train_conns)
    conns.append(conn(scale(ones(NUM_STATE, NUM_APL), WEIGHT_EXC)))       # 4 kc-apl
    conns.append(conn(scale(ones(NUM_APL, NUM_STATE), WEIGHT_INH_TEST)))  # 5 apl-kc
    conns.append(conn(scale(ones(NUM_DA, NUM_APL), WEIGHT_INH_TEST)))     # 6 da-apl
    conns.append(conn(scale(ones(NUM_APL, NUM_DA), WEIGHT_INH_TEST)))     # 7 apl-da
    conns.append(conn(scale(ones(NUM_DA, NUM_DA), WEIGHT_EXC)))           # 8 input-da
    conns.append(conn(scale(ones(NUM_DA, NUM_ACTION), WEIGHT_EXC)))       # 9 da-mbon
    return conns


# ---------------------------------------------------------------- 刺激
GT = [0.0, 0.8, 0.0, 1.0, 0.0]   # 绿色正立 T（安全）
BT = [0.0, 0.0, 0.8, 0.0, 1.0]   # 蓝色倒立 T（危险）


def stim_diff(a, b):
    return relu([x - y for x, y in zip(a, b)])


def conflict_input(c):
    """冲突刺激：绿色 (安全色, 浓度 c) + 倒立 T (危险形状, 固定 0.5)"""
    gt = [0.0, c, 0.0, 0.0, 0.5]
    bt = [0.0, 0.0, c, 0.5, 0.0]
    return stim_diff(gt, bt)


# ---------------------------------------------------------------- 训练
def train(verbose=False):
    net = DroDMTrainNet(build_train_conns())
    # 阶段 1: 学习 GT 安全 -> 动作 0 列被惩罚（r[:, 0] = -1）
    # 阶段 2: 学习 Bt 危险 -> 动作 1 列被惩罚（r[:, 1] = -1）
    for punished_action, stim in ((0, stim_diff(GT, BT)), (1, stim_diff(BT, GT))):
        net.reset()
        weight_trace = [[0.0] * NUM_ACTION for _ in range(NUM_STATE)]
        for _ in range(TRAIN_ROUNDS):
            out, dw_kc, dw_mbon = net.forward(stim)
            net.update(1, dw_kc)  # vis-kc 普通 STDP
            for k in range(NUM_STATE):
                for j in range(NUM_ACTION):
                    weight_trace[k][j] = weight_trace[k][j] * TRACE_DECAY + dw_mbon[k][j]
            if max(out) > 0.0:  # kc-mbon 奖励调制 R-STDP
                dw_r = [[(-1.0 if j == punished_action else 1.0) * weight_trace[k][j]
                         for j in range(NUM_ACTION)] for k in range(NUM_STATE)]
                net.update(2, dw_r)
                if verbose:
                    print("  训练输出:", out)
    return net


# ---------------------------------------------------------------- 测试
def pi_curve_linear(train_conns):
    t1, t2 = [], []
    for c in range(TEST_NUM):
        net = DroDMTrainNet(copy.deepcopy(train_conns))
        count = [0.0, 0.0]
        stim = conflict_input(c * 0.1)
        for _ in range(TEST_STEPS):
            out, _, _ = net.forward(stim)
            count[0] += out[0]
            count[1] += out[1]
        t1.append(count[0])
        t2.append(count[1])
    return t1, t2


def pi_curve_nonlinear(train_conns):
    t1, t2 = [], []
    for c in range(TEST_NUM):
        net = DroDMTestNet(build_test_conns(train_conns))
        count = [0.0, 0.0]
        stim = conflict_input(c * 0.1)
        for step in range(TEST_STEPS):
            input_da = [0.5] if (step < 10 and step % 2 == 0) else [0.0]
            out, dw_apl_kc, dw_kc_apl = net.forward(stim, input_da)
            update_weight(net.conns[5], dw_apl_kc, dim=0)  # apl-kc 可塑
            update_weight(net.conns[4], dw_kc_apl, dim=0)  # kc-apl 可塑
            count[0] += out[0]
            count[1] += out[1]
        t1.append(count[0])
        t2.append(count[1])
    return t1, t2


def to_pi(t1, t2):
    pi = []
    for a, b in zip(t1, t2):
        pi.append((a - b) / (a + b) if a + b > 0 else 0.0)
    return pi


def plot(x, p1, p2, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(x, p1, label="linear")
    plt.plot(x, p2, label="non-linear")
    plt.xlabel("color intensity")
    plt.ylabel("PI")
    plt.legend()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------- main
def main(plot_fig=False):
    print("=" * 60)
    print("果蝇蘑菇体冲突决策 SNN（BrainCog 范式，零依赖复刻）")
    print("=" * 60)
    print("\n[1/3] 训练：GT 绿色正立T=安全 / Bt 蓝色倒立T=危险 (20 轮 × 2)")
    net = train(verbose=False)
    print("训练后 KC-MBON 权重（行=KC, 列=动作0/动作1）:")
    for row in net.conns[2]["w"]:
        print("   ", ["%+.4f" % v for v in row])

    print("\n[2/3] 线性测试：冲突刺激下扫颜色浓度 c = 0 ~ 1.1")
    t1, t2 = pi_curve_linear(net.conns)
    p1 = to_pi(t1, t2)
    print("  t1 =", [int(v) for v in t1])
    print("  t2 =", [int(v) for v in t2])
    print("  PI =", ["%+.3f" % v for v in p1])

    print("\n[3/3] 非线性测试：+ APL 反馈抑制 + DA 多巴胺脉冲")
    t1b, t2b = pi_curve_nonlinear(net.conns)
    p2 = to_pi(t1b, t2b)
    print("  t1 =", [int(v) for v in t1b])
    print("  t2 =", [int(v) for v in t2b])
    print("  PI =", ["%+.3f" % v for v in p2])

    if plot_fig:
        try:
            import os
            os.makedirs("figures", exist_ok=True)
            plot([c * 0.1 for c in range(TEST_NUM)], p1, p2,
                 "figures/drosophila_pi.png")
            print("\n图已保存: figures/drosophila_pi.png")
        except ImportError:
            print("\n(matplotlib 未安装，跳过绘图；核心结果已在上方输出)")
    return p1, p2


if __name__ == "__main__":
    main(plot_fig="--plot" in sys.argv)
