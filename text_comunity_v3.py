# text_comunity_v3.py - 优化版完全异步流水线
import queue
import threading
import time
import sys
from base_interface import AudioData, TextData
from audio_player import AudioDriver
import control
from realtime_tts_processor import RealtimeTTSProcessor
from tts_driver import GenieTTSModule
import traceback

def init_all_modules():
    """初始化所有核心模块"""
    print("🔄 正在初始化所有模块...")
    
    # 1. 初始化LLM模块
    control.init_control_modules()
    print("✅ LLM模块初始化完成")
    
    # 2. 初始化音频驱动
    try:
        audio_driver = AudioDriver()
        audio_driver.start_play()
        print("✅ 音频播放模块初始化完成")
    except Exception as e:
        print(f"❌ 音频播放模块初始化失败: {e}")
        audio_driver = None
    
    # 3. 初始化TTS模块
    try:
        tts_module = GenieTTSModule()
        print("✅ TTS模块初始化完成")
    except Exception as e:
        print(f"❌ TTS模块初始化失败: {e}")
        return None, None, None
    
    return audio_driver, tts_module

def create_stream_pipeline(text_input, audio_driver, tts_module):
    """
    创建流式处理流水线（修复音频播放问题）
    """
    print(f"\n🚀 启动流式处理: '{text_input[:50]}...'")
    
    # 创建通信队列
    llm_to_tts_queue = queue.Queue(maxsize=20)    # LLM → TTS
    tts_to_audio_queue = queue.Queue(maxsize=30)  # TTS → Audio
    
    # 状态标志
    pipeline_status = {
        "llm_complete": False,
        "tts_complete": False,
        "audio_complete": False,
        "error": None,
        "tts_processor": None
    }
    
    # 事件信号
    first_audio_received = threading.Event()
    
    # 线程1: LLM生成线程
    def llm_thread():
        """LLM流式生成文本，发送到TTS队列"""
        try:
            ##print("🧠 LLM线程启动")
            
            # 直接调用control中的text_to_llm函数
            response = control.text_to_llm(text_input, llm_to_tts_queue)
            
            ##print(f"✅ LLM生成完成，共生成文本: {len(response)} 字符")
            pipeline_status["llm_complete"] = True
            
            # 等待TTS处理完成
            time.sleep(1)  # 给TTS一些时间处理最后的分片
            
        except Exception as e:
            print(f"❌ LLM线程错误: {e}")
            traceback.print_exc()
            pipeline_status["error"] = f"LLM错误: {e}"
    
    # 线程2: TTS处理线程
    def tts_thread():
        """TTS实时合成音频"""
        try:
            ##print("🔊 TTS线程启动")
            
            # 创建TTS处理器
            from realtime_tts_processor import RealtimeTTSProcessor
            tts_processor = RealtimeTTSProcessor(tts_module)
            pipeline_status["tts_processor"] = tts_processor
            
            # 启动实时处理
            tts_processor.start_processing(llm_to_tts_queue, tts_to_audio_queue)
            
            # 等待LLM完成
            while not pipeline_status["llm_complete"] and pipeline_status["error"] is None:
                time.sleep(0.1)
            
            # 等待一段时间，确保TTS处理完所有队列内容
            wait_count = 0
            while not llm_to_tts_queue.empty():
                wait_count += 1
                ##print(f"⏳ 等待TTS处理队列: {llm_to_tts_queue.qsize()} 项剩余")
                if wait_count > 60:  # 最多等待60秒
                    print("⚠️ TTS队列处理超时，强制继续")
                    break
                time.sleep(1)
            
            # 额外等待5秒，确保TTS完成当前合成
            ##print("⏳ 等待TTS完成最后合成...")
            for i in range(5):
                if pipeline_status["tts_processor"].is_running:
                    time.sleep(1)
                else:
                    break
            
            # 发送结束标记到音频队列
            tts_to_audio_queue.put(AudioData(
                pcm_data=b"",
                sample_rate=tts_module.sample_rate,
                channels=tts_module.channels,
                bit_depth=tts_module.bit_depth,
                is_finish=True
            ))
            
            pipeline_status["tts_complete"] = True
            print("✅ TTS线程完成")
            
        except Exception as e:
            print(f"❌ TTS线程错误: {e}")
            traceback.print_exc()
            pipeline_status["error"] = f"TTS错误: {e}"
    
    # 线程3: 音频播放线程（修复版本）
    def audio_thread():
        """音频实时播放（修复版本）"""
        try:
            ##print("🎵 音频播放线程启动")
            
            audio_chunk_count = 0
            start_time = time.time()
            first_chunk_time = None
            last_audio_time = time.time()
            
            # 重要：等待第一个音频分片（耐心等待TTS处理）
             ##print("⏳ 等待第一个音频分片...")
            initial_wait_time = 30  # 初始等待30秒
            wait_start = time.time()
            
            while time.time() - wait_start < initial_wait_time:
                try:
                    # 尝试获取音频数据，但设置较短超时
                    audio_data = tts_to_audio_queue.get(timeout=1.0)
                    
                    # 收到数据，继续处理
                    break
                    
                except queue.Empty:
                    # 检查是否TTS已完成且队列为空
                    if pipeline_status["tts_complete"] and tts_to_audio_queue.empty():
                        print("⚠️ TTS已完成但无音频数据")
                        return
                    
                    # 继续等待
                    elapsed = time.time() - wait_start
                     ##print(f"⏳ 等待第一个音频分片: {elapsed:.1f}/{initial_wait_time}秒")
                    continue
            
            # 处理第一个音频分片
            if audio_data.pcm_data == b"" and audio_data.is_finish:
                print("⚠️ 第一个分片就是结束标记")
                return
            
            # 播放第一个分片
            audio_chunk_count += 1
            first_chunk_time = time.time()
            first_chunk_latency = first_chunk_time - start_time
            
            # 计算分片信息
            chunk_size = len(audio_data.pcm_data)
            if hasattr(audio_data, 'sample_rate') and audio_data.sample_rate > 0:
                bytes_per_sample = audio_data.bit_depth // 8 if hasattr(audio_data, 'bit_depth') else 2
                channels = audio_data.channels if hasattr(audio_data, 'channels') else 1
                samples = chunk_size / (bytes_per_sample * channels)
                duration_ms = (samples / audio_data.sample_rate) * 1000
                duration_str = f", 时长: {duration_ms:.0f}ms"
            else:
                duration_str = ""
            
            ##print(f"⚡ 首音频分片延迟: {first_chunk_latency:.2f}秒")
            ##print(f"🎵 播放音频分片 #{audio_chunk_count}, 大小: {chunk_size}字节{duration_str}")
            
            # 播放第一个分片
            if audio_driver:
                audio_driver.push_audio_for_play(audio_data)
            
            last_audio_time = time.time()
            
            # 继续处理剩余音频分片
            no_audio_timeout = 10.0  # 10秒无音频超时
            
            while True:
                try:
                    # 获取音频数据
                    audio_data = tts_to_audio_queue.get(timeout=2.0)
                    
                    # 检查结束标记
                    if audio_data.pcm_data == b"" and audio_data.is_finish:
                        ##print("🎵 收到音频结束标记")
                        break
                    
                    # 播放音频
                    audio_chunk_count += 1
                    chunk_size = len(audio_data.pcm_data)
                    
                    if hasattr(audio_data, 'sample_rate') and audio_data.sample_rate > 0:
                        bytes_per_sample = audio_data.bit_depth // 8 if hasattr(audio_data, 'bit_depth') else 2
                        channels = audio_data.channels if hasattr(audio_data, 'channels') else 1
                        samples = chunk_size / (bytes_per_sample * channels)
                        duration_ms = (samples / audio_data.sample_rate) * 1000
                        duration_str = f", 时长: {duration_ms:.0f}ms"
                    else:
                        duration_str = ""
                    
                    ##print(f"🎵 播放音频分片 #{audio_chunk_count}, 大小: {chunk_size}字节{duration_str}")
                    
                    # 发送到音频驱动播放
                    if audio_driver:
                        audio_driver.push_audio_for_play(audio_data)
                    
                    last_audio_time = time.time()
                    
                except queue.Empty:
                    # 检查是否应该退出
                    if pipeline_status["tts_complete"] and tts_to_audio_queue.empty():
                        print("✅ TTS已完成且音频队列为空")
                        break
                    
                    # 检查是否超时无音频
                    if time.time() - last_audio_time > no_audio_timeout:
                        print(f"⚠️ {no_audio_timeout}秒无新音频，但继续等待")
                        # 不立即退出，而是继续等待
                        last_audio_time = time.time()
                    
                    continue
                
                except Exception as e:
                    print(f"❌ 音频播放错误: {e}")
                    traceback.print_exc()
                    break
            
            end_time = time.time()
            total_time = end_time - start_time
            
            ##print(f"✅ 音频播放完成，共播放 {audio_chunk_count} 个音频分片")
            ##print(f"⏱️  总处理时间: {total_time:.2f}秒")
            
            if audio_chunk_count > 0 and first_chunk_time:
                useful_time = end_time - first_chunk_time
                ##print(f"📊 有效音频时间: {useful_time:.2f}秒")
            
            pipeline_status["audio_complete"] = True
            
        except Exception as e:
            print(f"❌ 音频线程错误: {e}")
            traceback.print_exc()
            pipeline_status["error"] = f"音频错误: {e}"
    
    # 启动所有线程
    start_time = time.time()
    
    # 创建线程
    llm_t = threading.Thread(target=llm_thread, name="LLM-Gen")
    tts_t = threading.Thread(target=tts_thread, name="TTS-Synth")
    audio_t = threading.Thread(target=audio_thread, name="Audio-Play")
    
    # 设置线程属性
    llm_t.daemon = False
    tts_t.daemon = False
    audio_t.daemon = False
    
    # 启动线程（注意启动顺序）
    print("🔄 启动处理线程...")
    
    # 先启动TTS和音频线程
    tts_t.start()
    time.sleep(1)  # 确保TTS线程启动
    
    audio_t.start()
    time.sleep(0.5)  # 确保音频线程启动
    
    # 最后启动LLM线程
    llm_t.start()
    
    # 等待线程完成（按依赖顺序）
    ##print("⏳ 等待LLM线程完成...")
    llm_t.join(timeout=120)
    
    ##print("⏳ 等待TTS线程完成...")
    tts_t.join(timeout=90)
    
    ##print("⏳ 等待音频线程完成...")
    audio_t.join(timeout=60)
    
    # 检查错误
    if pipeline_status["error"]:
        print(f"❌ 处理过程中出错: {pipeline_status['error']}")
        return None
    
    # 检查完成状态
    end_time = time.time()
    total_time = end_time - start_time
    
    ##print(f"\n✅ 流式处理完成，总耗时: {total_time:.2f}秒")
    ##print(f"📊 状态: LLM={pipeline_status['llm_complete']}, "
          ##f"TTS={pipeline_status['tts_complete']}, "
          ##f"Audio={pipeline_status['audio_complete']}")
    
    return True

def main():
    """主函数：文本对话系统"""
    print("\n" + "="*60)
    print("🚀 文本对话系统 - 完全异步流式版")
    print("="*60)
    print("📌 特性：")
    print("  1. LLM流式生成文本")
    print("  2. TTS实时合成音频")
    print("  3. 音频实时播放")
    print("  4. 三线程并行处理，极低延迟")
    print("  5. 记忆系统支持")
    print("  6. 智能句子分割")
    print("="*60)
    print("📝 命令：")
    print("  - 输入对话文本进行交流")
    print("  - 输入 'exit' 或 'quit' 退出程序")
    print("  - 输入 'clear' 清空对话历史")
    print("  - 输入 'help' 显示帮助")
    print("="*60)
    
    # 初始化所有模块
    audio_driver, tts_module = init_all_modules()
    
    if not audio_driver or not tts_module:
        print("❌ 模块初始化失败，程序退出")
        return
    
    conversation_count = 0
    
    try:
        while True:
            # 获取用户输入
            try:
                user_text = input("\n👤 你: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\n⚠️ 输入中断，退出程序")
                break
            
            # 处理特殊命令
            if user_text.lower() in ['exit', 'quit']:
                print("\n👋 退出程序...")
                break
            elif user_text.lower() == 'clear':
                print("🧹 清空对话历史")
                # TODO: 添加对话历史清空功能
                continue
            elif user_text.lower() == 'help':
                print("\n📋 帮助信息:")
                print("  直接输入文本进行对话")
                print("  'exit'/'quit': 退出程序")
                print("  'clear': 清空对话历史")
                print("  'help': 显示帮助")
                continue
            elif not user_text:
                print("⚠️ 输入不能为空")
                continue
            
            # 处理对话
            conversation_count += 1
            print(f"\n🔄 第 {conversation_count} 轮对话开始...")
            
            # 创建并运行流水线
            result = create_stream_pipeline(user_text, audio_driver, tts_module)
            
            if result:
                print(f"✅ 第 {conversation_count} 轮对话完成")
            else:
                print(f"❌ 第 {conversation_count} 轮对话失败")
            
            # 短暂休息，让系统稳定
            time.sleep(0.5)
    
    except Exception as e:
        print(f"\n❌ 程序运行出错：{e}")
        traceback.print_exc()
    
    finally:
        # 释放所有资源
        print("\n🧹 正在清理资源...")
        
        # 停止控制模块
        control.is_running = False
        control.cleanup()
        
        # 释放音频驱动
        if audio_driver:
            try:
                audio_driver.release()
                print("✅ 音频驱动已释放")
            except:
                pass
        
        # 清理TTS模块
        if tts_module:
            try:
                # 调用TTS模块的清理方法
                tts_module.__del__()
                print("✅ TTS模块已清理")
            except:
                pass
        
        print("✅ 所有资源已释放，程序退出")

def test_single_input():
    """测试单次输入"""
    print("🧪 测试模式：单次输入")
    
    # 初始化所有模块
    audio_driver, tts_module = init_all_modules()
    
    if not audio_driver or not tts_module:
        print("❌ 模块初始化失败")
        return
    
    # 测试输入
    test_inputs = [
        "你好，介绍一下你自己",
        "今天的天气怎么样？",
        "讲一个有趣的故事",
        "1+1等于几？"
    ]
    
    print(f"\n📋 可用测试输入:")
    for i, test_text in enumerate(test_inputs, 1):
        print(f"  {i}. {test_text}")
    
    try:
        choice = input("\n请选择测试输入 (1-4): ").strip()
        if choice in ['1', '2', '3', '4']:
            test_text = test_inputs[int(choice)-1]
        else:
            test_text = input("或输入自定义测试文本: ").strip()
        
        if not test_text:
            test_text = "你好，介绍一下你自己"
        
        print(f"\n🔄 开始测试: '{test_text}'")
        
        # 创建并运行流水线
        result = create_stream_pipeline(test_text, audio_driver, tts_module)
        
        if result:
            print("✅ 测试完成")
        else:
            print("❌ 测试失败")
    
    except Exception as e:
        print(f"❌ 测试出错: {e}")
        traceback.print_exc()
    
    finally:
        # 清理资源
        print("\n🧹 清理测试资源...")
        control.is_running = False
        control.cleanup()
        if audio_driver:
            audio_driver.release()

if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_single_input()
    else:
        main()