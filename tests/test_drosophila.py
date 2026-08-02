# -*- coding: utf-8 -*-
"""果蝇蘑菇体决策范式测试（纯标准库 unittest，零依赖）。"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from drosophila_decision import (
    NUM_ACTION, NUM_STATE, TEST_NUM, TRAIN_ROUNDS,
    DroDMTrainNet, build_train_conns, conflict_input,
    main, pi_curve_linear, pi_curve_nonlinear, stim_diff, to_pi, train,
    GT, BT,
)


class TestStimuli(unittest.TestCase):
    def test_training_stimuli(self):
        # GT-Bt 正部 = [绿0.8, 正立T1.0]，Bt-GT 正部 = [蓝0.8, 倒立t1.0]
        self.assertEqual(stim_diff(GT, BT), [0.0, 0.8, 0.0, 1.0, 0.0])
        self.assertEqual(stim_diff(BT, GT), [0.0, 0.0, 0.8, 0.0, 1.0])

    def test_conflict_input(self):
        # 冲突刺激 = 安全色(绿, 浓度c) + 危险形状(倒立T, 0.5)
        self.assertEqual(conflict_input(0.3), [0.0, 0.3, 0.0, 0.0, 0.5])
        self.assertEqual(conflict_input(0.0), [0.0, 0.0, 0.0, 0.0, 0.5])


class TestTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.net = train()

    def test_kc_mbon_rows_normalized(self):
        # 每轮 update 后做 L1 行归一，绝对值行和应恒为 1
        for row in self.net.conns[2]["w"]:
            self.assertAlmostEqual(sum(abs(v) for v in row), 1.0, places=6)

    def test_learned_association(self):
        # 绿 KC(行1)/正立T KC(行3) 应驱动动作1；蓝 KC(行2)/倒立T KC(行4) 驱动动作0
        w = self.net.conns[2]["w"]
        self.assertGreater(w[1][1], w[1][0])
        self.assertGreater(w[3][1], w[3][0])
        self.assertGreater(w[2][0], w[2][1])
        self.assertGreater(w[4][0], w[4][1])

    def test_deterministic(self):
        net2 = train()
        for a, b in zip(self.net.conns[2]["w"], net2.conns[2]["w"]):
            self.assertEqual(a, b)


class TestPICurves(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        net = train()
        cls.conns = net.conns

    def test_shape_and_range(self):
        t1, t2 = pi_curve_linear(self.conns)
        pi = to_pi(t1, t2)
        self.assertEqual(len(pi), TEST_NUM)
        for v in pi:
            self.assertGreaterEqual(v, -1.0)
            self.assertLessEqual(v, 1.0)

    def test_decision_flip(self):
        # 低浓度：危险形状主导 -> 动作0 多 (PI>0)；高浓度：安全色主导 -> PI<0
        t1, t2 = pi_curve_linear(self.conns)
        self.assertGreater(t1[0], t2[0])
        self.assertGreater(t2[-1], t1[-1])

    def test_nonlinear_sharper(self):
        # 非线性版本应出现接近全有/全无的开关（极端浓度下 |PI| > 线性版）
        p1 = to_pi(*pi_curve_linear(self.conns))
        p2 = to_pi(*pi_curve_nonlinear(self.conns))
        self.assertGreater(abs(p2[1]), abs(p1[1]))
        self.assertGreater(abs(p2[-2]), abs(p1[-2]))
        # 转折后非线性应接近饱和 -1
        self.assertLess(p2[6], -0.9)

    def test_main_runs(self):
        p1, p2 = main(plot_fig=False)
        self.assertEqual(len(p1), TEST_NUM)
        self.assertEqual(len(p2), TEST_NUM)


if __name__ == "__main__":
    unittest.main()
