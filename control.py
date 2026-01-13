# control.py 开头修改导入
import gc
import torch
import queue
import threading
import time
from typing import Optional
from base_interface import AudioData, TextData, ChatHistory
from llm_zhipu_driver import init_model_and_tokenizer, CUSTOM_SYSTEM_PROMPT, create_stream_generator
import keyboard  # 需安装：pip install keyboard
from sentence_processor import SentenceProcessor
name="妮可(Nicole)"
# ===================== 全局控制标记 =====================
is_recording: bool = False  # 是否激活语音识别
is_running: bool = True     # 程序是否运行
asr_input_q: Optional[queue.Queue] = None    # 全局ASR输入队列

# ===================== 模块实例（延迟初始化） =====================
tokenizer = None
llm_model = None

# ===================== 初始化函数 =====================
def memory_cleanup():
    """清理显存和内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
def init_control_modules():
    """初始化LLM相关模块（供外部调用）"""
    global tokenizer, llm_model
    tokenizer, llm_model = init_model_and_tokenizer()
    print("✅ 控制模块（LLM）初始化完成")

# ===================== 桥接函数封装 =====================
def asr_to_llm(asr_output_q: queue.Queue, tts_input_q: queue.Queue):
    """ASR → 句子处理器 → LLM"""
    # 使用自定义system prompt初始化对话历史
    chat_history = [{"role": "system", "content": CUSTOM_SYSTEM_PROMPT}]
    sentence_processor = SentenceProcessor(min_length=3, max_silence=1.5)
    sentence_queue = queue.Queue()  # 存储完整句子
    
    # 启动句子处理线程
    def process_asr_output():
        while is_running:
            try:
                asr_text = asr_output_q.get(timeout=0.1)
                sentence_processor.process(asr_text, sentence_queue)
            except queue.Empty:
                continue
    
    # 启动LLM处理线程
    def process_sentences():
        nonlocal chat_history  # 声明为nonlocal变量
        while is_running:
            try:
                sentence_data = sentence_queue.get(timeout=0.1)
                if not sentence_data.text:
                    continue
                
                # 调用LLM
                print(f"\n👤 用户说: {sentence_data.text}")
                print("="*50)
                
                if llm_model is None or tokenizer is None:
                    continue
                
                print(f"🤖 {name}: ", end="", flush=True)
                full_response = ""
                
                # 使用流式生成器
                for chunk, new_history in create_stream_generator(
                    tokenizer=tokenizer,
                    model=llm_model,
                    query=sentence_data.text,
                    history=chat_history
                ):
                    if chunk:
                        print(chunk, end="", flush=True)
                        full_response += chunk
                        tts_input_q.put(TextData(text=chunk, is_finish=False))
                
                # 更新对话历史
                chat_history = new_history if new_history else chat_history
                
                # 发送结束标记
                tts_input_q.put(TextData(text="", is_finish=True))
                print(f"\n{'='*50}")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"\n❌ LLM处理错误: {e}")
                import traceback
                traceback.print_exc()
                # 发送错误提示给TTS
                error_text = "抱歉，我遇到了一些问题，请稍后再试。"
                tts_input_q.put(TextData(text=error_text, is_finish=True))
                tts_input_q.put(TextData(text="", is_finish=True))
                continue
    
    # 启动两个线程
    asr_thread = threading.Thread(target=process_asr_output, name="ASR句子处理")
    llm_thread = threading.Thread(target=process_sentences, name="LLM处理")
    
    asr_thread.daemon = True
    llm_thread.daemon = True
    
    asr_thread.start()
    llm_thread.start()
    
    # 等待线程结束
    try:
        while is_running:
            time.sleep(0.1)
    finally:
        # 等待线程结束
        asr_thread.join(timeout=1)
        llm_thread.join(timeout=1)
    def process_sentences():
        nonlocal chat_history
        while is_running:
            try:
                # 每次处理前清理内存
                memory_cleanup()
                
                sentence_data = sentence_queue.get(timeout=0.1)
                if not sentence_data.text:
                    continue
                
                # 处理完成后再次清理
                # ...
                
                # 清理历史记录，只保留最近几轮
                if len(chat_history) > 10:  # 限制对话历史长度
                    chat_history = [chat_history[0]] + chat_history[-8:]  # 保留系统提示和最近对话
                    
            except queue.Empty:
                continue
            finally:
                # 确保清理
                memory_cleanup()


def tts_to_play(tts_output_q: queue.Queue, audio_driver):
    """
    TTS合成结果 → 音频播放（修复多次播放逻辑，支持音频格式透传）
    :param tts_output_q: TTS输出队列（AudioData）
    :param audio_driver: 音频驱动实例（AudioDriver）
    """
    audio_chunk_count = 0
    
    while is_running:
        # 1. 读取TTS合成的音频分片（非阻塞）
        try:
            audio_data: AudioData = tts_output_q.get(timeout=0.1)
        except queue.Empty:
            continue
        
        # 2. 处理结束标记：仅推送空数据，不终止循环
        if audio_data.pcm_data == b"":
            # 推送结束标记（重置播放流）
            audio_driver.push_audio_for_play(audio_data)
            print(f"🎵 收到TTS结束标记（第{audio_chunk_count}个分片后）")
            audio_chunk_count = 0  # 重置计数
            continue
        
        # 3. 调用音频驱动播放接口（透传TTS的原始音频格式）
        audio_chunk_count += 1
        chunk_size = len(audio_data.pcm_data)
        
        # 计算音频时长
        if hasattr(audio_data, 'sample_rate') and audio_data.sample_rate > 0:
            # 假设是16位PCM（2字节/样本）
            bytes_per_sample = audio_data.bit_depth // 8 if hasattr(audio_data, 'bit_depth') else 2
            channels = audio_data.channels if hasattr(audio_data, 'channels') else 1
            samples = chunk_size / (bytes_per_sample * channels)
            duration_ms = (samples / audio_data.sample_rate) * 1000
            duration_str = f", 时长≈{duration_ms:.1f}ms"
        else:
            duration_str = ""
        
        print(f"🎵 推送音频分片 #{audio_chunk_count}, "
              f"大小: {chunk_size} 字节{duration_str}")
        
        audio_driver.push_audio_for_play(audio_data)

def key_control(audio_driver):
    """
    按键控制线程：空格=启动/停止识别，ESC=退出程序
    增加防抖和状态检查，避免阻塞
    """
    global is_recording, is_running, asr_input_q
    
    print("="*50)
    print("🎙️  流式语音交互系统")
    print("→ 按【空格键】：开始/停止语音输入")
    print("→ 按【ESC键】：退出程序")
    print("="*50)
    
    last_space_press = 0
    debounce_time = 0.5  # 防抖时间500ms
    
    while is_running:
        try:
            current_time = time.time()
            
            # 空格键：切换录音状态（带防抖）
            if keyboard.is_pressed('space') and (current_time - last_space_press) > debounce_time:
                last_space_press = current_time
                is_recording = not is_recording
                if is_recording:
                    print("\n▶️  已启动语音识别，开始说话...")
                    audio_driver.start_record(chunk_duration=0.6)
                else:
                    print("\n⏹️  已停止语音识别，正在处理结果...")
                    audio_driver.stop_record()
                    # 发送ASR结束标记
                    if asr_input_q is not None:
                        asr_input_q.put(AudioData(pcm_data=b"", sample_rate=16000, channels=1, is_finish=True))
            
            # ESC键：退出程序
            if keyboard.is_pressed('esc'):
                print("\n🛑 收到退出指令，正在关闭系统...")
                is_running = False
                is_recording = False
                break
            
            time.sleep(0.05)  # 稍微降低轮询频率
            
        except Exception as e:
            print(f"⌨️  按键控制异常: {e}")
            break
    
    print("⌨️  按键控制线程退出")

# ===================== 资源清理函数 =====================
def cleanup():
    """清理全局状态和资源"""
    global is_running, is_recording, asr_input_q
    is_running = False
    is_recording = False
    asr_input_q = None
    print("✅ 控制模块资源已清理")