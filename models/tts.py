# -*- coding: utf-8 -*-
"""
语音合成模块（TTS）
使用 Windows 内置 SAPI 语音合成
"""
import subprocess
import os
import tempfile


def text_to_speech(text: str, output_path: str = None,
                   voice: str = None, rate: int = 0) -> str:
    """文本转语音

    Args:
        text: 要合成的文本
        output_path: 输出音频文件路径（wav格式），None则自动生成
        voice: 语音名称，None使用默认
        rate: 语速 -10~10，0为正常

    Returns:
        输出文件路径
    """
    if output_path is None:
        output_path = os.path.join(tempfile.gettempdir(), f"tts_{os.getpid()}.wav")

    # 用 PowerShell 调用 System.Speech
    ps_script = f"""
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
{('$synth.SelectVoice("' + voice + '")') if voice else ''}
$synth.Rate = {rate}
$synth.SetOutputToWaveFile("{output_path}")
$synth.Speak(@'
{text}
'@)
$synth.Dispose()
"""

    try:
        result = subprocess.run(
            ["powershell", "-Command", ps_script],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0 and os.path.exists(output_path):
            return output_path
        else:
            print(f"TTS 错误: {result.stderr}")
            return None
    except Exception as e:
        print(f"TTS 异常: {e}")
        return None


def speak(text: str, voice: str = None, rate: int = 0) -> bool:
    """直接播放语音（不保存文件）

    Args:
        text: 要合成的文本
        voice: 语音名称
        rate: 语速

    Returns:
        是否成功
    """
    ps_script = f"""
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
{('$synth.SelectVoice("' + voice + '")') if voice else ''}
$synth.Rate = {rate}
$synth.Speak(@'
{text}
'@)
$synth.Dispose()
"""

    try:
        result = subprocess.run(
            ["powershell", "-Command", ps_script],
            capture_output=True, text=True, timeout=60
        )
        return result.returncode == 0
    except Exception as e:
        print(f"播放异常: {e}")
        return False


def list_voices() -> list:
    """列出可用语音"""
    ps_script = """
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.GetInstalledVoices() | ForEach-Object {
    Write-Output $_.VoiceInfo.Name
}
$synth.Dispose()
"""

    try:
        result = subprocess.run(
            ["powershell", "-Command", ps_script],
            capture_output=True, text=True, timeout=10
        )
        voices = [line.strip() for line in result.stdout.strip().split('\n')
                  if line.strip()]
        return voices
    except Exception:
        return []


if __name__ == "__main__":
    # 测试
    print("可用语音:")
    for v in list_voices():
        print(f"  - {v}")

    print()
    print("测试语音合成...")
    output = text_to_speech("你好，我是AI大脑，很高兴认识你！")
    if output:
        print(f"生成成功: {output}")
    else:
        print("生成失败")
