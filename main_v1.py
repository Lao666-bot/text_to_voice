#!/usr/bin/env python3
"""
语音交互系统主程序（修复版）
实现：录音 → ASR识别 → 句子整合 → LLM处理 → TTS合成 → 音频播放
"""
from memory_manager import memory_manager, monitor_memory, cleanup_memory
import queue
import threading
import time
import signal
import sys
import os
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入各个模块
from audio_player import AudioDriver
from funasr_driver import FunASRStreamingASR
from tts_driver import GenieTTSModule
from control import init_control_modules, asr_to_llm, tts_to_play, key_control, cleanup, is_running as control_running, asr_input_q
from base_interface import AudioData, TextData
from sentence_processor import SentenceProcessor

# ===================== 全局变量 =====================
# 队列定义
asr_input_queue = None      # 音频 → ASR（音频数据）
asr_output_queue = None     # ASR → LLM（文本数据）
tts_input_queue = None      # LLM → TTS（文本数据）
tts_output_queue = None     # TTS → 播放（音频数据）

# 模块实例
audio_driver = None
asr_module = None
tts_module = None

# 线程控制
threads = []
should_stop = threading.Event()

# ===================== 初始化函数 =====================
def init_modules():
    """初始化所有模块"""
    global audio_driver, asr_module, tts_module
    global asr_input_queue, asr_output_queue, tts_input_queue, tts_output_queue
    
    print("=" * 60)
    print("🚀 语音交互系统启动中...")
    print("=" * 60)
    
    try:
        # 1. 初始化音频驱动
        print("[1/6] 初始化音频驱动...")
        audio_driver = AudioDriver()
        time.sleep(0.5)
        
        # 2. 初始化ASR模块
        print("[2/6] 初始化ASR模块...")
        asr_module = FunASRStreamingASR()
        time.sleep(0.5)
        
        # 3. 初始化TTS模块
        print("[3/6] 初始化TTS模块...")
        tts_module = GenieTTSModule()
        time.sleep(0.5)
        
        # 4. 初始化LLM控制模块
        print("[4/6] 初始化LLM模块...")
        init_control_modules()
        time.sleep(1)
        
        # 5. 创建队列
        print("[5/6] 创建数据队列...")
        asr_input_queue = queue.Queue(maxsize=100)
        asr_output_queue = queue.Queue(maxsize=50)
        tts_input_queue = queue.Queue(maxsize=50)
        tts_output_queue = queue.Queue(maxsize=100)
        
        # 6. 设置全局队列引用
        from control import asr_input_q as global_asr_input_q
        global_asr_input_q = asr_input_queue
        
        print("[6/6] 系统初始化完成！")
        print("=" * 60)
        print("🎯 系统已就绪，等待指令...")
        print("→ 按【空格键】开始/停止录音")
        print("→ 按【ESC键】退出系统")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

# ===================== 音频采集到ASR的桥接 =====================
def audio_to_asr():
    """音频采集 → ASR识别"""
    print("🎤 音频-ASR桥接线程启动")
    
    while not should_stop.is_set():
        try:
            # 从音频驱动获取录音数据（非阻塞）
            try:
                audio_data = audio_driver.get_record_queue().get(timeout=0.1)
            except queue.Empty:
                continue
            
            # 处理结束标记
            if audio_data.pcm_data == b"" and audio_data.is_finish:
                print("📝 ASR接收到录音结束标记")
                if asr_input_queue is not None:
                    asr_input_queue.put(audio_data)  # 传递结束标记
                continue
            
            # 将音频数据推送给ASR
            if asr_input_queue is not None and not should_stop.is_set():
                asr_input_queue.put(audio_data)
                
        except Exception as e:
            if not should_stop.is_set():
                print(f"❌ 音频-ASR桥接错误: {e}")
            continue
    
    print("🎤 音频-ASR桥接线程退出")

# ===================== ASR处理线程 =====================
def asr_processing_thread():
    """ASR处理线程"""
    print("🔤 ASR处理线程启动")
    
    try:
        # 启动ASR流式处理
        asr_module.stream_process(asr_input_queue, asr_output_queue)
    except Exception as e:
        print(f"❌ ASR处理线程异常: {e}")
        import traceback
        traceback.print_exc()
    
    print("🔤 ASR处理线程退出")

# ===================== TTS处理线程 =====================
def tts_processing_thread():
    """TTS处理线程"""
    print("🗣️  TTS处理线程启动")
    
    try:
        # 启动TTS流式处理
        tts_module.stream_process(tts_input_queue, tts_output_queue)
    except Exception as e:
        print(f"❌ TTS处理线程异常: {e}")
        import traceback
        traceback.print_exc()
    
    print("🗣️  TTS处理线程退出")

# ===================== ASR到LLM桥接线程 =====================
def asr_to_llm_thread():
    """ASR → LLM 桥接线程"""
    print("🧠 ASR-LLM桥接线程启动")
    
    try:
        # 调用控制模块的asr_to_llm函数
        asr_to_llm(asr_output_queue, tts_input_queue)
    except Exception as e:
        print(f"❌ ASR-LLM桥接线程异常: {e}")
        import traceback
        traceback.print_exc()
    
    print("🧠 ASR-LLM桥接线程退出")

# ===================== TTS到播放桥接线程 =====================
def tts_to_play_thread():
    """TTS → 播放 桥接线程"""
    print("🎵 TTS-播放桥接线程启动")
    
    try:
        # 调用控制模块的tts_to_play函数
        tts_to_play(tts_output_queue, audio_driver)
    except Exception as e:
        print(f"❌ TTS-播放桥接线程异常: {e}")
        import traceback
        traceback.print_exc()
    
    print("🎵 TTS-播放桥接线程退出")

# ===================== 按键控制线程 =====================
def key_control_thread():
    """按键控制线程"""
    print("⌨️  按键控制线程启动")
    
    try:
        # 调用控制模块的key_control函数
        key_control(audio_driver)
    except Exception as e:
        print(f"❌ 按键控制线程异常: {e}")
        import traceback
        traceback.print_exc()
    
    print("⌨️  按键控制线程退出")

# ===================== 信号处理 =====================
def signal_handler(signum, frame):
    """处理退出信号"""
    print(f"\n📶 收到信号 {signum}，正在退出...")
    should_stop.set()
    
    # 通知控制模块停止运行
    from control import is_running as control_is_running
    control_is_running = False

# ===================== 清理函数 =====================
def cleanup_resources():
    """清理所有资源"""
    global audio_driver, asr_module, tts_module
    
    print("\n🧹 正在清理资源...")
    
    # 停止所有线程
    should_stop.set()
    
    # 等待线程结束
    for thread in threads:
        if thread.is_alive():
            thread.join(timeout=1)
    
    # 清理控制模块
    cleanup()
    
    # 停止音频驱动
    if audio_driver:
        try:
            audio_driver.stop_record()
            audio_driver.stop_play()
            audio_driver.release()
        except:
            pass
    
    # 清理TTS模块
    if tts_module:
        try:
            tts_module.__del__()
        except:
            pass
    
    # 清空队列
    for q in [asr_input_queue, asr_output_queue, tts_input_queue, tts_output_queue]:
        if q:
            try:
                while not q.empty():
                    try:
                        q.get_nowait()
                    except:
                        break
            except:
                pass
    
    print("✅ 所有资源已清理")
    print("👋 系统退出")

# ===================== 线程监控函数 =====================
def monitor_threads():
    """监控线程状态"""
    while not should_stop.is_set():
        alive_count = sum(1 for t in threads if t.is_alive())
        print(f"📊 线程状态: {alive_count}/{len(threads)} 个线程运行中")
        time.sleep(5)

# ===================== 主函数 =====================
def main():
    """主函数"""
    global audio_driver, threads
    monitor_memory()
    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # 初始化模块
        if not init_modules():
            print("❌ 初始化失败，系统退出")
            return
        
        # 启动音频播放（常驻）
        audio_driver.start_play()
        
        # 创建线程列表
        threads = []
        
        # 创建线程
        thread_functions = [
            (audio_to_asr, "音频-ASR桥接"),
            (asr_processing_thread, "ASR处理"),
            (asr_to_llm_thread, "ASR-LLM桥接"),
            (tts_processing_thread, "TTS处理"),
            (tts_to_play_thread, "TTS-播放桥接"),
            (key_control_thread, "按键控制")
        ]
        
        # 启动所有线程
        for func, name in thread_functions:
            thread = threading.Thread(target=func, name=name)
            thread.daemon = True
            threads.append(thread)
            thread.start()
            print(f"✅ 启动线程: {name}")
            time.sleep(0.2)  # 稍微错开启动时间
        
        # 启动线程监控
        monitor_thread = threading.Thread(target=monitor_threads, name="线程监控")
        monitor_thread.daemon = True
        threads.append(monitor_thread)
        monitor_thread.start()
        
        print(f"✅ 共启动 {len(threads)} 个线程")
        print("=" * 60)
        
        # 主线程等待（直到收到退出信号）
        try:
            while not should_stop.is_set():
                time.sleep(0.5)
                
                # 检查控制模块的运行状态
                from control import is_running as control_is_running
                if not control_is_running:
                    should_stop.set()
                    break
                    
        except KeyboardInterrupt:
            print("\n👆 收到键盘中断信号")
            should_stop.set()
        
        # 清理资源
        cleanup_resources()
        
    except Exception as e:
        print(f"❌ 主程序异常: {e}")
        import traceback
        traceback.print_exc()
        cleanup_resources()
        raise
    # 在主循环中添加定期清理
    try:
            while not should_stop.is_set():
                time.sleep(5)
                
                # 每5秒检查一次内存
                cleanup_memory()
                
                # 检查控制模块的运行状态
                from control import is_running as control_is_running
                if not control_is_running:
                    should_stop.set()
                    break
                    
    except KeyboardInterrupt:
            print("\n👆 收到键盘中断信号")
            should_stop.set()
        
        # 清理资源
            cleanup_resources()
        
        
    finally:
        # 停止内存监控
        memory_manager.stop_monitoring()

# ===================== 简化的测试函数 =====================
def test_flow():
    """测试流程：简化版本"""
    print("🧪 测试流程启动...")
    
    try:
        # 初始化
        audio_driver = AudioDriver()
        asr_module = FunASRStreamingASR()
        tts_module = GenieTTSModule()
        init_control_modules()
        
        # 启动音频播放
        audio_driver.start_play()
        
        # 创建队列
        asr_input_q = queue.Queue()
        asr_output_q = queue.Queue()
        tts_input_q = queue.Queue()
        tts_output_q = queue.Queue()
        
        # 设置全局引用
        from control import asr_input_q as global_asr_input_q
        global_asr_input_q = asr_input_q
        
        # 启动关键线程
        threading.Thread(
            target=asr_module.stream_process,
            args=(asr_input_q, asr_output_q),
            daemon=True
        ).start()
        
        threading.Thread(
            target=asr_to_llm,
            args=(asr_output_q, tts_input_q),
            daemon=True
        ).start()
        
        threading.Thread(
            target=tts_module.stream_process,
            args=(tts_input_q, tts_output_q),
            daemon=True
        ).start()
        
        threading.Thread(
            target=tts_to_play,
            args=(tts_output_q, audio_driver),
            daemon=True
        ).start()
        
        # 启动按键控制（阻塞）
        print("🎯 按空格开始录音，ESC退出")
        key_control(audio_driver)
        
    except Exception as e:
        print(f"❌ 测试流程异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'audio_driver' in locals():
            audio_driver.release()

# ===================== 执行入口 =====================
if __name__ == "__main__":
    print("""
    ███████╗██╗   ██╗███████╗████████╗███████╗███╗   ███╗
    ██╔════╝██║   ██║██╔════╝╚══██╔══╝██╔════╝████╗ ████║
    ███████╗██║   ██║█████╗     ██║   █████╗  ██╔████╔██║
    ╚════██║██║   ██║██╔══╝     ██║   ██╔══╝  ██║╚██╔╝██║
    ███████║╚██████╔╝██║        ██║   ███████╗██║ ╚═╝ ██║
    ╚══════╝ ╚═════╝ ╚═╝        ╚═╝   ╚══════╝╚═╝     ╚═╝
    语音交互系统 v1.0
    """)
    
    # 两种启动方式：
    # 1. 完整模式（推荐）
    main()
    
    # 2. 测试模式（简化）
    # test_flow()