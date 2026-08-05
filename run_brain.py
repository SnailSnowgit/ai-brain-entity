# -*- coding: utf-8 -*-
"""
AI Brain 交互式运行脚本

用法：
    python run_brain.py

功能：
    - 与大脑进行对话
    - 测试语言理解功能
    - 查看大脑状态
    - 观察思考活动
"""
import sys
import os

# 确保能导入项目模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai_brain_entity import AIBrainEntity


def print_banner():
    """打印欢迎横幅"""
    print("=" * 60)
    print("  🧠 AI Brain - 类脑智能体交互控制台")
    print("=" * 60)
    print()
    print("  命令列表：")
    print("    直接输入文本  - 与大脑对话（Qwen2生成回复）")
    print("    /understand T - 仅理解文本，不生成回复")
    print("    /think N     - 让大脑思考 N 步")
    print("    /introspect  - 内省：查看大脑自我意识")
    print("    /status      - 查看大脑完整状态")
    print("    /memory      - 查看记忆内容")
    print("    /thoughts    - 查看当前思考空间")
    print("    /stream N    - 意识流 N 步")
    print("    /help        - 显示帮助")
    print("    /quit        - 退出")
    print()
    print("=" * 60)
    print()


def show_status(brain):
    """显示大脑状态"""
    print()
    print("📊 大脑状态：")
    print("-" * 40)
    status = brain.status()
    print(status)
    print()


def show_thoughts(brain):
    """显示思考空间"""
    print()
    print("💭 思考空间：")
    print("-" * 40)
    if brain.thought_space:
        for i, t in enumerate(sorted(brain.thought_space,
                                     key=lambda x: x.activation, reverse=True)):
            bar = "█" * int(t.activation * 20)
            print(f"  {i+1}. [{t.source:10s}] {bar} {t.activation:.2f}")
            print(f"      {t.content[:50]}")
    else:
        print("  （空）")
    print()


def show_memory(brain):
    """显示记忆"""
    print()
    print("📝 记忆状态：")
    print("-" * 40)
    print(f"  短期记忆 (STM): {len(brain.short_term_memory)}/{brain.stm_capacity}")
    print(f"  长期记忆 (LTM): {len(brain.long_term_memory)}/{brain.ltm_capacity}")
    print()
    if brain.long_term_memory:
        print("  长期记忆（前5条）：")
        for i, m in enumerate(brain.long_term_memory[:5]):
            print(f"    {i+1}. [{m.tag}] {m.content[:40]}... (w={m.weight:.2f})")
    print()


def do_think(brain, steps):
    """让大脑思考"""
    print()
    print(f"🤔 思考 {steps} 步...")
    print("-" * 40)
    for i in range(steps):
        result = brain.think()
        thought = result.get("thought", "")
        recalled = result.get("recalled", [])
        consolidated = result.get("consolidated", [])
        print(f"  第 {i+1} 步: {thought[:50]}...")
        if recalled:
            print(f"    联想: {', '.join([r[:20] for r in recalled[:3]])}")
        if consolidated:
            print(f"    固化: {len(consolidated)} 条记忆")
    print()


def do_introspect(brain):
    """内省"""
    print()
    print("🔍 内省 - 自我感知：")
    print("-" * 40)
    result = brain.introspect(depth="deep")
    print(f"  情绪: {result.get('mood', 'unknown')}")
    print(f"  当前念头: {result.get('top_thought', 'none')}")
    print(f"  新奇度: {result.get('novelty', 0):.3f}")
    print(f"  注意力: {result.get('attention', 0):.3f}")
    print(f"  思考空间: {result.get('thought_space_size', 0)} 个念头")
    print()
    print(f"  💬 内省言语:")
    print(f"    {result.get('text', '')}")
    print()


def do_stream(brain, steps):
    """意识流"""
    print()
    print(f"🌊 意识流 {steps} 步...")
    print("-" * 40)
    result = brain.stream_of_consciousness(steps=steps, daydream=0.3)
    chain = result.get("chain", [])
    for i, thought in enumerate(chain):
        print(f"  {i+1}. {thought[:50]}...")
    print()
    insights = result.get("insights", [])
    if insights:
        print(f"  💡 灵感闪现:")
        for insight in insights:
            print(f"    - {insight[:50]}...")
    print()


def do_understand(brain, text):
    """理解文本"""
    print()
    print(f"📖 理解: \"{text}\"")
    print("-" * 40)
    result = brain.understand(text)
    print(f"  编码器: {result['meta'].get('encoder', 'unknown')}")
    print(f"  新奇度: {result['novelty']:.3f}")
    print(f"  思考空间: {result['thought_space_size']} 个念头")

    emotion = result.get("emotion", {})
    dominant = max(emotion, key=emotion.get) if emotion else "unknown"
    print(f"  主导情绪: {dominant} ({emotion.get(dominant, 0):.3f})")

    cross_modal = result.get("cross_modal_recalled", [])
    if cross_modal:
        print(f"  跨模态联想: {', '.join(cross_modal)}")
    print()


def do_reply(brain, text):
    """对话回复"""
    print()
    print(f"你: \"{text}\"")
    print("-" * 40)
    print("🧠 思考中...")
    result = brain.reply(text, think_ticks=2)

    reply = result.get("reply", "")
    mood = result.get("dominant_mood", "unknown")
    novelty = result.get("novelty", 0)

    print()
    print(f"大脑: {reply}")
    print()
    print(f"  情绪: {mood}")
    print(f"  新奇度: {novelty:.3f}")
    print(f"  思考空间: {result['thought_space_size']} 个念头")

    recalled = result.get("recalled", [])
    if recalled:
        print(f"  联想记忆: {', '.join([r[:20] for r in recalled[:3]])}")

    gen_meta = result.get("gen_meta", {})
    if gen_meta:
        print(f"  生成器: {gen_meta.get('generator', 'unknown')}")
    print()


def main():
    """主交互循环"""
    print_banner()

    # 创建大脑
    print("🧠 正在初始化大脑...")
    brain = AIBrainEntity("InteractiveBrain")
    print("✅ 大脑初始化完成！")
    print()

    # 主循环
    while True:
        try:
            user_input = input("你> ").strip()

            if not user_input:
                continue

            # 命令处理
            if user_input.startswith("/"):
                parts = user_input.split()
                cmd = parts[0].lower()

                if cmd == "/quit" or cmd == "/exit":
                    print("👋 再见！")
                    break

                elif cmd == "/help":
                    print_banner()

                elif cmd == "/status":
                    show_status(brain)

                elif cmd == "/thoughts":
                    show_thoughts(brain)

                elif cmd == "/memory":
                    show_memory(brain)

                elif cmd == "/think":
                    steps = int(parts[1]) if len(parts) > 1 else 3
                    do_think(brain, steps)

                elif cmd == "/introspect":
                    do_introspect(brain)

                elif cmd == "/stream":
                    steps = int(parts[1]) if len(parts) > 1 else 5
                    do_stream(brain, steps)

                elif cmd == "/understand":
                    text = " ".join(parts[1:]) if len(parts) > 1 else ""
                    if text:
                        do_understand(brain, text)
                    else:
                        print("❓ 用法: /understand <文本>")
                        print()

                else:
                    print(f"❓ 未知命令: {cmd}")
                    print("   输入 /help 查看帮助")
                    print()

            else:
                # 普通文本：与大脑对话
                do_reply(brain, user_input)

        except KeyboardInterrupt:
            print()
            print("👋 再见！")
            break

        except Exception as e:
            print(f"❌ 错误: {e}")
            print()


if __name__ == "__main__":
    main()
