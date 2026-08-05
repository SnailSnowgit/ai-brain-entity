# -*- coding: utf-8 -*-
"""
AI Brain 语音对话脚本

用法：
    python run_voice_chat.py

功能：
    - 文本输入，语音输出
    - 音频文件输入（语音识别）
    - 完整的语音对话体验
"""
import sys
import os

# 确保能导入项目模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai_brain_entity import AIBrainEntity


def print_banner():
    """打印欢迎横幅"""
    print("=" * 60)
    print("  🎙️ AI Brain - 语音对话控制台")
    print("=" * 60)
    print()
    print("  命令列表：")
    print("    直接输入文本  - 与大脑对话（语音回复）")
    print("    /audio <文件> - 从音频文件输入（语音识别）")
    print("    /speak <文本> - 直接测试语音合成")
    print("    /voices       - 列出可用语音")
    print("    /voice <名称> - 切换语音")
    print("    /rate <速度>  - 设置语速 (-10~10)")
    print("    /status       - 查看大脑状态")
    print("    /thoughts     - 查看思考空间")
    print("    /help         - 显示帮助")
    print("    /quit         - 退出")
    print()
    print("=" * 60)
    print()


def speak_text(text, voice=None, rate=0):
    """播放语音"""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))
    from tts import speak
    return speak(text, voice=voice, rate=rate)


def list_voices():
    """列出可用语音"""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))
    from tts import list_voices
    return list_voices()


def audio_to_text(audio_path):
    """音频转文字（Whisper）"""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models", "encoders"))
    from multimodal import encode_audio
    result = encode_audio(audio_path)
    return result.get("text", "")


def do_voice_reply(brain, text, voice=None, rate=0):
    """语音对话回复"""
    print()
    print(f"你: \"{text}\"")
    print("-" * 40)
    print("🧠 思考中...")

    result = brain.reply(text, think_ticks=2)
    reply = result.get("reply", "")
    mood = result.get("dominant_mood", "unknown")

    print()
    print(f"大脑: {reply}")
    print()
    print(f"  情绪: {mood}")
    print(f"  新奇度: {result['novelty']:.3f}")
    print(f"  思考空间: {result['thought_space_size']} 个念头")

    # 语音播放
    if reply:
        print()
        print("🔊 播放语音...")
        success = speak_text(reply, voice=voice, rate=rate)
        if success:
            print("  ✅ 播放完成")
        else:
            print("  ❌ 播放失败")
    print()


def main():
    """主交互循环"""
    print_banner()

    # 创建大脑
    print("🧠 正在初始化大脑...")
    brain = AIBrainEntity("VoiceBrain")
    print("✅ 大脑初始化完成！")

    # 语音设置
    current_voice = "Microsoft Huihui Desktop"
    current_rate = 0

    print(f"🎙️ 当前语音: {current_voice}")
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

                elif cmd == "/voices":
                    voices = list_voices()
                    print()
                    print("🎤 可用语音:")
                    for v in voices:
                        marker = "  ← 当前" if v == current_voice else ""
                        print(f"  - {v}{marker}")
                    print()

                elif cmd == "/voice":
                    if len(parts) > 1:
                        voice_name = " ".join(parts[1:])
                        voices = list_voices()
                        if voice_name in voices:
                            current_voice = voice_name
                            print(f"✅ 已切换到语音: {voice_name}")
                        else:
                            print(f"❌ 找不到语音: {voice_name}")
                            print("   输入 /voices 查看可用语音")
                    else:
                        print("❓ 用法: /voice <语音名称>")
                    print()

                elif cmd == "/rate":
                    if len(parts) > 1:
                        try:
                            rate = int(parts[1])
                            if -10 <= rate <= 10:
                                current_rate = rate
                                print(f"✅ 语速已设置为: {rate}")
                            else:
                                print("❌ 语速范围: -10 ~ 10")
                        except ValueError:
                            print("❌ 请输入数字")
                    else:
                        print(f"当前语速: {current_rate}")
                    print()

                elif cmd == "/speak":
                    text = " ".join(parts[1:]) if len(parts) > 1 else ""
                    if text:
                        print(f"🔊 播放: {text}")
                        speak_text(text, voice=current_voice, rate=current_rate)
                    else:
                        print("❓ 用法: /speak <文本>")
                    print()

                elif cmd == "/audio":
                    audio_path = " ".join(parts[1:]) if len(parts) > 1 else ""
                    if audio_path and os.path.exists(audio_path):
                        print()
                        print(f"🎵 识别音频: {audio_path}")
                        print("-" * 40)
                        text = audio_to_text(audio_path)
                        print(f"识别结果: {text}")
                        if text:
                            do_voice_reply(brain, text,
                                          voice=current_voice, rate=current_rate)
                    else:
                        print("❓ 用法: /audio <音频文件路径>")
                        print("   支持 wav/mp3 等格式")
                    print()

                elif cmd == "/status":
                    print()
                    print("📊 大脑状态：")
                    print("-" * 40)
                    print(brain.status())
                    print()

                elif cmd == "/thoughts":
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

                else:
                    print(f"❓ 未知命令: {cmd}")
                    print("   输入 /help 查看帮助")
                    print()

            else:
                # 普通文本：语音对话
                do_voice_reply(brain, user_input,
                              voice=current_voice, rate=current_rate)

        except KeyboardInterrupt:
            print()
            print("👋 再见！")
            break

        except Exception as e:
            print(f"❌ 错误: {e}")
            print()


if __name__ == "__main__":
    main()
