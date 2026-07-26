# -*- coding: utf-8 -*-
"""
观测台数据统一导出入口：一条命令刷新「ai_brain 大脑观测台」
两个 Widget 的全部回放数据。纯本地，无任何外部 API。

  python export_widget_data.py

等价于依次运行：
  python brain_activity_trace.py      —— 大脑活动监视器（帧级神经状态回放）
  python thought_chain_scenarios.py   —— 脉冲思考链 · Spike CoT（三场景对照）
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import brain_activity_trace
import thought_chain_scenarios


def main() -> None:
    print("[1/2] 大脑活动监视器数据 ...")
    brain_activity_trace.main()
    print("[2/2] 脉冲思考链（Spike CoT）数据 ...")
    thought_chain_scenarios.main()
    print("观测台数据已全部刷新。")


if __name__ == "__main__":
    main()
