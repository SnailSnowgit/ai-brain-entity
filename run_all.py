# -*- coding: utf-8 -*-
"""
全项目统一运行脚本：按依赖顺序一键运行 ai_brain 全部脚本，
验证"核心 → 群体/测试 → 实验 → 观测数据"整条关联链。纯本地，无外部 API。

  python run_all.py                 # 完整流程（含实验 1-8，约 2-3 分钟）
  python run_all.py --quick         # 跳过实验与演示，只做测试 + 观测数据刷新
  python run_all.py --skip-demo     # 跳过两个演示脚本

各步骤与脚本关联图的对应：
  [1] tests/                    核心层行为回归（改核心后必须先过）
  [2] ai_brain_entity.py        核心层内置演示（冒烟）
  [3] swarm.py                  群体层演示（依赖核心层）
  [4] experiments.py            实验层：实验 1-8 → figures/ + experiment_results.json
  [5] export_widget_data.py     观测层：刷新观测台两个 Widget 的数据
任一步失败即中断并报告。
"""
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).parent
PY = sys.executable


def step(title: str, cmd: list) -> float:
    print(f"\n===== {title} =====", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=BASE)
    dt = time.time() - t0
    if r.returncode != 0:
        print(f"\n[失败] {title}（退出码 {r.returncode}），流程中断。")
        sys.exit(r.returncode)
    print(f"[完成] {title}（{dt:.1f}s）")
    return dt


def main() -> None:
    quick = "--quick" in sys.argv
    skip_demo = "--skip-demo" in sys.argv or quick
    skip_exp = "--skip-experiments" in sys.argv or quick

    total = 0.0
    total += step("1/5 单元测试（核心层回归，16 项）",
                  [PY, "-m", "unittest", "discover", "tests"])
    if not skip_demo:
        total += step("2/5 核心层内置演示（ai_brain_entity.py）",
                      [PY, "ai_brain_entity.py"])
        total += step("3/5 群体层演示（swarm.py）", [PY, "swarm.py"])
    else:
        print("\n[跳过] 演示脚本（--skip-demo/--quick）")
    if not skip_exp:
        total += step("4/5 实验 1-8 复现（experiments.py）",
                      [PY, "experiments.py"])
    else:
        print("\n[跳过] 实验复现（--skip-experiments/--quick）")
    total += step("5/5 观测台数据刷新（export_widget_data.py）",
                  [PY, "export_widget_data.py"])

    print(f"\n全部完成，总耗时 {total:.1f}s。")
    print("产物：data/experiment_results.json、figures/exp*.png、"
          "data/brain_activity_trace.json、data/thought_chain_scenarios.json")


if __name__ == "__main__":
    main()
