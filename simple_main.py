# simple_main.py
import queue
import threading
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from audio_player import AudioDriver
from funasr_driver import FunASRStreamingASR
from control import init_control_modules, asr_to_llm, tts_to_play, key_control, cleanup

def simple_main():
    """简化版主程序，减少模块和线程"""
    print("🚀 启动简化版语音交互系统")
    
    # 初始化关键模块
    audio_driver = AudioDriver()
    asr_module = FunASRStreamingASR()
    init_control_modules()
    
    # 启动音频播放
    audio_driver.start_play()
    
    # 创建队列（缩小尺寸）
    asr_input_q = queue.Queue(maxsize=10)
    asr_output_q = queue.Queue(maxsize=5)
    tts_input_q = queue.Queue(maxsize=5)
    tts_output_q = queue.Queue(maxsize=10)
    
    # 设置全局引用
    from control import asr_input_q as global_asr_input_q
    global_asr_input_q = asr_input_q
    
    # 启动核心线程（减少线程数）
    threads = []
    
    # ASR处理线程
    asr_thread = threading.Thread(
        target=asr_module.stream_process,
        args=(asr_input_q, asr_output_q),
        name="ASR处理",
        daemon=True
    )
    threads.append(asr_thread)
    
    # ASR到LLM桥接
    bridge_thread = threading.Thread(
        target=asr_to_llm,
        args=(asr_output_q, tts_input_q),
        name="桥接处理",
        daemon=True
    )
    threads.append(bridge_thread)
    
    # 注意：这里移除了独立的TTS线程，改为在bridge_thread中处理
    
    # TTS到播放桥接
    play_thread = threading.Thread(
        target=tts_to_play,
        args=(tts_output_q, audio_driver),
        name="播放处理",
        daemon=True
    )
    threads.append(play_thread)
    
    # 启动所有线程
    for thread in threads:
        thread.start()
        time.sleep(0.5)
    
    print("✅ 系统已就绪，按空格开始录音，ESC退出")
    
    # 直接运行按键控制（阻塞）
    try:
        key_control(audio_driver)
    except KeyboardInterrupt:
        print("\n👆 收到中断信号")
    finally:
        # 清理
        cleanup()
        audio_driver.release()
        print("👋 系统退出")

if __name__ == "__main__":
    simple_main()