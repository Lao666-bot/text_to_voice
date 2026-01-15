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
from llm_zhipu_driver import MemorySystem, create_stream_generator
import random
import llm_zhipu_driver
from memory_database import MemoryDatabase
from llm_zhipu_driver import DatabaseMemorySystem
from memory_adapter import MemoryAdapter
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
    global tokenizer, llm_model, memory_adapter
    
    tokenizer, llm_model, _ = init_model_and_tokenizer()
    
    # 创建记忆适配器
    memory_adapter = MemoryAdapter(llm_model, tokenizer)
    
    print("✅ 控制模块（增强记忆版）初始化完成")

# ===================== 桥接函数封装 =====================
# control.py - 修改asr_to_llm函数，集成记忆系统

# 在control.py开头添加

# 修改asr_to_llm函数
def asr_to_llm(asr_output_q: queue.Queue, tts_input_q: queue.Queue):
    """ASR → 句子处理器 → LLM（增强记忆版）"""
    # 初始化记忆适配器
    memory_adapter = MemoryAdapter(llm_model, tokenizer)
    
    sentence_processor = SentenceProcessor(min_length=3, max_silence=1.5)
    sentence_queue = queue.Queue()
    
    # 启动句子处理线程
    def process_asr_output():
        while is_running:
            try:
                asr_text = asr_output_q.get(timeout=0.1)
                sentence_processor.process(asr_text, sentence_queue)
            except queue.Empty:
                continue
    
    # 启动LLM处理线程（增强记忆版）
    def process_sentences():
        """LLM处理线程（增强记忆版），支持句子缓冲"""
        # 初始化记忆适配器
        memory_adapter = MemoryAdapter(llm_model, tokenizer)
        
        sentence_processor = SentenceProcessor(min_length=3, max_silence=1.5)
        sentence_queue = queue.Queue()
        
        # 句子缓冲区：用于累积LLM生成的文本
        tts_buffer = ""
        # 句子结束标记
        sentence_end_markers = ['。', '！', '？', '.', '!', '?', '\n']
        # 最大缓冲区长度（字符数）
        max_buffer_length = 100
        
        while is_running:
            try:
                sentence_data = sentence_queue.get(timeout=0.1)
                if not sentence_data.text:
                    continue
                
                user_input = sentence_data.text
                
                print(f"\n👤 用户说: {user_input}")
                print("="*50)
                
                # 使用记忆适配器处理查询
                print(f"🤖 {name}: ", end="", flush=True)
                full_response = ""
                
                # 检查是否应该强制使用记忆
                force_memory = False
                memory_keywords = ['之前', '刚才', '记得', '说过', '告诉过']
                if any(keyword in user_input for keyword in memory_keywords):
                    force_memory = True
                    print("🧠 检测到记忆关键词，强制使用记忆...")
                
                # 获取流式响应
                response_generator = memory_adapter.process_query_stream(
                    user_input, 
                    use_memory=True,
                    force_memory=force_memory,
                    temperature=0.2
                )
                
                # 处理流式响应
                for chunk, is_final in response_generator:
                    if chunk:
                        full_response += chunk
                        print(chunk, end="", flush=True)
                        
                        # 将chunk添加到缓冲区
                        tts_buffer += chunk
                        
                        # 检查是否需要发送缓冲区到TTS
                        should_send = False
                        
                        # 条件1：遇到句子结束标记
                        if any(marker in tts_buffer for marker in sentence_end_markers):
                            # 找到最后一个句子结束标记的位置
                            last_end_pos = max([tts_buffer.rfind(marker) for marker in sentence_end_markers 
                                            if tts_buffer.rfind(marker) >= 0], default=-1)
                            
                            if last_end_pos >= 0:
                                # 发送到句子结束标记为止的内容
                                to_send = tts_buffer[:last_end_pos+1]
                                tts_input_q.put(TextData(text=to_send, is_finish=False))
                                # 保留剩余部分
                                tts_buffer = tts_buffer[last_end_pos+1:]
                        
                        # 条件2：缓冲区达到最大长度
                        elif len(tts_buffer) >= max_buffer_length:
                            # 尽量在标点处分割
                            split_pos = -1
                            for marker in [',', '，', ';', '；', '、']:
                                pos = tts_buffer.rfind(marker)
                                if pos > split_pos:
                                    split_pos = pos
                            
                            if split_pos > 0:
                                to_send = tts_buffer[:split_pos+1]
                                tts_buffer = tts_buffer[split_pos+1:]
                            else:
                                to_send = tts_buffer
                                tts_buffer = ""
                            
                            tts_input_q.put(TextData(text=to_send, is_finish=False))
                    
                    if is_final:
                        # 发送剩余内容
                        if tts_buffer.strip():
                            tts_input_q.put(TextData(text=tts_buffer, is_finish=False))
                        # 发送结束标记
                        tts_input_q.put(TextData(text="", is_finish=True))
                        tts_buffer = ""
                        break
                
                # 显示统计信息
                stats = memory_adapter.get_stats()
                print(f"\n📊 记忆统计: {stats['memory_hits']}/{stats['conversation_count']}次使用记忆")
                
                print(f"\n{'='*50}")
                
                # 内存清理
                memory_cleanup()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"\n❌ LLM处理错误: {e}")
                import traceback
                traceback.print_exc()
                error_text = "抱歉，我刚才有点走神了，我们继续聊吧。"
                tts_input_q.put(TextData(text=error_text, is_finish=True))
                continue

def load_memory():
    """从文件加载记忆"""
    try:
        import json
        import os
        
        if os.path.exists("memory_backup.json"):
            with open("memory_backup.json", "r", encoding="utf-8") as f:
                memory_data = json.load(f)
            
            memory_system = MemorySystem()
            memory_system.long_term_memory = memory_data.get("long_term_memory", [])
            memory_system.user_profile = memory_data.get("user_profile", {})
            print("✅ 记忆已从文件加载")
            return memory_system
    except Exception as e:
        print(f"❌ 记忆加载失败: {e}")
    
    return MemorySystem()  # 返回新的记忆系统


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
            ##print(f"🎵 收到TTS结束标记（第{audio_chunk_count}个分片后）")
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