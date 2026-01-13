# text_comunity_v2.py（优化版 - 支持 LLM 流式输出时实时 TTS）
import queue
import threading
import time
from base_interface import AudioData, TextData
from audio_player import AudioDriver
import control
from sentence_processor import SentenceProcessor

def init_all_modules():
    """初始化所有核心模块"""
    # 1. 初始化LLM模块
    control.init_control_modules()
    
    # 2. 初始化音频驱动
    audio_driver = AudioDriver()
    audio_driver.start_play()
    print("✅ 音频播放模块初始化完成")
    
    # 3. 初始化TTS模块
    from tts_driver import GenieTTSModule
    tts_module = GenieTTSModule()
    print("✅ TTS模块初始化完成")
    
    return audio_driver, tts_module

def stream_llm_to_tts(text_input, audio_driver, tts_module):
    """流式处理：LLM流式输出时实时进行TTS合成"""
    print(f"🚀 启动流式处理: LLM → TTS")
    
    # 创建队列
    llm_input_queue = queue.Queue()
    tts_input_queue = queue.Queue()
    tts_output_queue = queue.Queue()
    
    # 句子处理器：用于将LLM流式输出分割成完整句子
    sentence_processor = SentenceProcessor(min_length=3, max_silence=0.5)
    sentence_queue = queue.Queue()
    
    # 结果收集器（用于收集完整的LLM回复）
    full_response = ""
    
    # 线程1: LLM生成线程（流式）
    def llm_generator():
        """LLM流式生成"""
        nonlocal full_response
        
        # 构建对话历史
        chat_history = [{"role": "system", "content": control.CUSTOM_SYSTEM_PROMPT}]
        
        # 将用户输入放入队列
        llm_input_queue.put(TextData(text=text_input, is_finish=True))
        
        # 处理用户输入
        while True:
            try:
                input_data = llm_input_queue.get(timeout=0.1)
                if input_data.text:
                    print(f"\n👤 用户输入: {input_data.text}")
                    print("="*50)
                    
                    if control.llm_model is None or control.tokenizer is None:
                        print("❌ LLM模型未初始化")
                        break
                    
                    print(f"🤖 {control.name}: ", end="", flush=True)
                    
                    # 流式生成LLM回复
                    for chunk, new_history in control.create_stream_generator(
                        tokenizer=control.tokenizer,
                        model=control.llm_model,
                        query=input_data.text,
                        history=chat_history
                    ):
                        if chunk:
                            print(chunk, end="", flush=True)
                            full_response += chunk
                            
                            # 将每个chunk送入句子处理器
                            sentence_processor.process(
                                TextData(text=chunk, is_finish=False),
                                sentence_queue
                            )
                    
                    # 发送结束标记给句子处理器
                    sentence_processor.process(
                        TextData(text="", is_finish=True),
                        sentence_queue
                    )
                    
                    # 更新对话历史
                    chat_history = new_history if new_history else chat_history
                    print(f"\n{'='*50}")
                    break
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"\n❌ LLM生成错误: {e}")
                import traceback
                traceback.print_exc()
                break
    
    # 线程2: 句子处理线程（从句子队列中取出完整句子给TTS）
    def sentence_handler():
        """处理完整句子并发送给TTS"""
        sentence_count = 0
        
        while True:
            try:
                # 从句子队列获取完整句子
                sentence_data = sentence_queue.get(timeout=0.1)
                
                if not sentence_data.text:
                    if sentence_data.is_finish:
                        # 发送TTS结束标记
                        tts_input_queue.put(TextData(text="", is_finish=True))
                        ##print(f"📝 LLM生成完成，共处理{sentence_count}个句子")
                        break
                    continue
                
                sentence_count += 1
                print(f"📦 句子 #{sentence_count}: {sentence_data.text}")
                
                # 将完整句子发送给TTS
                tts_input_queue.put(TextData(text=sentence_data.text, is_finish=False))
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ 句子处理错误: {e}")
                break
    
    # 线程3: TTS合成线程
    def tts_generator():
        """TTS合成线程"""
        ##print("🔄 启动TTS合成线程...")
        tts_module.stream_process(tts_input_queue, tts_output_queue)
    
    # 线程4: 音频播放线程
    def audio_player():
        """音频播放线程"""
        ##print("🔄 启动音频播放线程...")
        control.tts_to_play(tts_output_queue, audio_driver)
    
    # 启动所有线程
    threads = []
    
    llm_thread = threading.Thread(target=llm_generator, name="LLM生成线程")
    sentence_thread = threading.Thread(target=sentence_handler, name="句子处理线程")
    tts_thread = threading.Thread(target=tts_generator, name="TTS合成线程")
    play_thread = threading.Thread(target=audio_player, name="音频播放线程")
    
    # 设置守护线程
    for thread in [llm_thread, sentence_thread, tts_thread, play_thread]:
        thread.daemon = True
        threads.append(thread)
        thread.start()
    
    # 等待关键线程完成
    llm_thread.join(timeout=60)
    sentence_thread.join(timeout=60)
    
    # 等待TTS和播放线程有足够时间处理
    print("⏳ 等待TTS处理完成...")
    time.sleep(2)
    
    # 检查TTS和播放线程是否还在运行
    tts_thread.join(timeout=10)
    play_thread.join(timeout=10)
    
    print(f"✅ 流式处理完成，完整回复:\n{full_response}")

def process_single_round(text_input, audio_driver, tts_module):
    """处理单轮对话（使用流式处理）"""
    ##print(f"\n🔄 开始处理: '{text_input[:50]}...'")
    start_time = time.time()
    
    # 使用流式处理
    stream_llm_to_tts(text_input, audio_driver, tts_module)
    
    end_time = time.time()
    print(f"✅ 本轮对话完成，耗时: {end_time - start_time:.2f}秒")

def main():
    """主流程：文本输入→LLM流式推理→实时TTS合成→音频播放"""
    print("\n" + "="*60)
    print("🚀 文本到语音转换系统（流式版）")
    print("="*60)
    print("📌 特性：")
    print("  1. LLM流式回复，逐字显示")
    print("  2. 实时句子分割，不等完整回复")
    print("  3. TTS流式音频合成（与LLM输出同步）")
    print("  4. 音频实时播放")
    print("  5. 输入'exit'退出程序")
    print("="*60)
    
    # 初始化所有模块
    audio_driver, tts_module = init_all_modules()
    
    try:
        while True:
            # 获取用户输入
            user_text = input("\n请输入对话文本：").strip()
            
            if user_text.lower() == "exit":
                print("\n👋 退出程序...")
                break
            
            if not user_text:
                print("⚠️ 输入不能为空！")
                continue
            
            # 处理单轮对话
            process_single_round(user_text, audio_driver, tts_module)
    
    except KeyboardInterrupt:
        print("\n\n⚠️ 程序被手动中断")
    except Exception as e:
        print(f"\n❌ 程序运行出错：{e}")
        import traceback
        traceback.print_exc()
    finally:
        # 释放所有资源
        print("\n🧹 正在清理资源...")
        audio_driver.release()
        control.is_running = False
        control.cleanup()
        print("✅ 所有资源已释放，程序退出")

if __name__ == "__main__":
    main()