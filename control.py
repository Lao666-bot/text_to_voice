# control.py
import gc
import torch
import queue
import threading
import time
from typing import Optional, List
from base_interface import AudioData, TextData, ChatHistory
from llm_zhipu_driver import init_model_and_tokenizer, CUSTOM_SYSTEM_PROMPT, MemorySystem
import keyboard
from sentence_processor import SentenceProcessor
import re

name = "妮可(Nicole)"

# ===================== 全局控制标记 =====================
is_recording: bool = False
is_running: bool = True
asr_input_q: Optional[queue.Queue] = None

# ===================== 模块实例 =====================
tokenizer = None
llm_model = None

# ===================== 智能句子分割器 =====================
class SmartSentenceSplitter:
    """实时智能分句器"""
    
    def __init__(self, min_chunk_length=5, max_chunk_length=100):
        self.min_length = min_chunk_length
        self.max_length = max_chunk_length
        self.buffer = ""
        
        # 句子结束符
        self.endings = ['。', '！', '？', '.', '!', '?', '；', ';']
        self.weak_endings = ['，', ',', '、', '：', ':']
        
    def add_text(self, text):
        """添加文本并返回完整的句子"""
        self.buffer += text
        sentences = []
        
        # 查找所有句子结束符的位置
        end_positions = []
        for ending in self.endings:
            pos = self.buffer.find(ending)
            while pos != -1:
                end_positions.append(pos)
                pos = self.buffer.find(ending, pos + 1)
        
        # 按位置排序
        end_positions.sort()
        
        # 提取完整的句子
        last_end = 0
        for pos in end_positions:
            sentence = self.buffer[last_end:pos+1].strip()
            if len(sentence) >= self.min_length:
                sentences.append(sentence)
                last_end = pos + 1
        
        # 更新缓冲区
        self.buffer = self.buffer[last_end:]
        
        # 检查缓冲区是否过长
        if len(self.buffer) > self.max_length:
            # 在弱结束符处分割
            split_pos = -1
            for ending in self.weak_endings:
                pos = self.buffer.rfind(ending)
                if pos > split_pos:
                    split_pos = pos
            
            if split_pos > 0:
                long_sentence = self.buffer[:split_pos+1]
                if long_sentence.strip():
                    sentences.append(long_sentence)
                self.buffer = self.buffer[split_pos+1:]
            else:
                # 直接分割
                long_sentence = self.buffer
                if long_sentence.strip():
                    sentences.append(long_sentence)
                self.buffer = ""
        
        return sentences
    
    def flush(self):
        """清空缓冲区，返回剩余内容"""
        remaining = self.buffer
        self.buffer = ""
        return remaining

# ===================== 初始化函数 =====================
def memory_cleanup():
    """清理显存和内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def init_control_modules():
    """初始化LLM相关模块"""
    global tokenizer, llm_model
    tokenizer, llm_model, _ = init_model_and_tokenizer()
    print("✅ 控制模块初始化完成")

# ===================== 真正的异步流式生成 =====================
def create_async_stream_generator(user_input, history=None, memory_system=None, temperature=0.8):
    """创建异步流式生成器"""
    
    # 准备记忆上下文
    memory_context = ""
    if memory_system:
        try:
            memory_context = memory_system.get_memory_context(user_input)
            if memory_context and "（暂无记忆）" not in memory_context:
                print(f"🧠 使用记忆: {memory_context[:50]}...")
        except Exception as e:
            print(f"⚠️ 记忆系统错误: {e}")
    
    # 构建动态提示词
    dynamic_prompt = CUSTOM_SYSTEM_PROMPT.format(
        memory_context=memory_context
    )
    
    # 准备对话历史
    if not history or history[0].get("role") != "system":
        history = [{"role": "system", "content": dynamic_prompt}]
    else:
        history[0]["content"] = dynamic_prompt
    
    ##print(f"🚀 LLM开始异步生成...")
    
    # 使用线程实现真正的异步
    result_queue = queue.Queue()
    
    def generate_stream():
        """在独立线程中生成流"""
        try:
            full_response = ""
            
            # 使用模型的stream_chat方法
            for response, new_history, _ in llm_model.stream_chat(
                tokenizer=tokenizer,
                query=user_input,
                history=history,
                top_p=0.9,
                temperature=temperature,
                system=dynamic_prompt,
                past_key_values=None,
                return_past_key_values=True
            ):
                # 过滤AI身份关键词
                filter_words = ["AI", "助手", "ChatGLM", "模型", "训练", "开发", "智谱", "人工智能"]
                filtered_response = response
                for word in filter_words:
                    filtered_response = filtered_response.replace(word, "")
                
                # 提取新增的内容
                if len(filtered_response) > len(full_response):
                    new_content = filtered_response[len(full_response):]
                    full_response = filtered_response
                    
                    if new_content:
                        # 将新内容放入队列
                        result_queue.put(("chunk", new_content))
            
            # 生成完成
            result_queue.put(("complete", full_response))
            
        except Exception as e:
            print(f"❌ LLM流式生成错误: {e}")
            result_queue.put(("error", str(e)))
    
    # 启动生成线程
    gen_thread = threading.Thread(target=generate_stream, daemon=True)
    gen_thread.start()
    
    # 返回一个生成器，从队列中读取结果
    while True:
        try:
            item_type, data = result_queue.get(timeout=30)  # 30秒超时
            
            if item_type == "chunk":
                yield data, False, ""
            elif item_type == "complete":
                yield "", True, data
                break
            elif item_type == "error":
                raise Exception(f"LLM生成错误: {data}")
                
        except queue.Empty:
            print("⏳ LLM生成超时")
            break

# ===================== 异步LLM-TTS流水线 =====================
def asr_to_llm(asr_output_q: queue.Queue, tts_input_q: queue.Queue):
    """ASR → LLM → TTS（真正的异步流水线）"""
    
    memory_system = MemorySystem()
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
    
    # LLM-TTS并行处理线程
    def process_conversation():
        while is_running:
            try:
                sentence_data = sentence_queue.get(timeout=0.1)
                if not sentence_data.text:
                    continue
                
                user_input = sentence_data.text
                
                print(f"\n👤 用户说: {user_input}")
                print("="*50)
                
                print(f"🤖 {name}: ", end="", flush=True)
                
                # 检查记忆关键词
                force_memory = any(keyword in user_input for keyword in ['之前', '刚才', '记得', '说过', '告诉过'])
                if force_memory:
                    print("🧠 检测到记忆关键词，强制使用记忆...")
                
                # 创建智能分句器
                sentence_splitter = SmartSentenceSplitter(min_length=3, max_length=40)
                
                # 开始异步流式生成
                start_time = time.time()
                tts_chunks_sent = 0
                full_response = ""
                
                # 使用异步流式生成器
                for chunk, is_final, final_response in create_async_stream_generator(
                    user_input,
                    memory_system=memory_system,
                    temperature=0.2 if force_memory else 0.8
                ):
                    if chunk:
                        # 打印chunk
                        print(chunk, end="", flush=True)
                        full_response += chunk
                        
                        # 添加到分句器
                        sentences = sentence_splitter.add_text(chunk)
                        
                        # 发送完整的句子到TTS
                        for sentence in sentences:
                            if sentence.strip():
                                tts_chunks_sent += 1
                                # print(f"\n📤 发送TTS分片#{tts_chunks_sent}: {sentence}")
                                tts_input_q.put(TextData(text=sentence, is_finish=False))
                    
                    if is_final:
                        # 发送剩余的文本
                        remaining = sentence_splitter.flush()
                        if remaining.strip():
                            tts_chunks_sent += 1
                            # print(f"\n📤 发送最终TTS分片#{tts_chunks_sent}: {remaining}")
                            tts_input_q.put(TextData(text=remaining, is_finish=False))
                        
                        # 发送结束标记
                        tts_input_q.put(TextData(text="", is_finish=True))
                        
                        # 更新记忆
                        memory_system.add_conversation(user_input, final_response)
                        
                        end_time = time.time()
                        print(f"\n⏱️  响应时间: {end_time - start_time:.2f}秒")
                        print(f"📤 共发送{tts_chunks_sent}个TTS分片")
                        break
                
                print(f"\n{'='*50}")
                
                # 内存清理
                memory_cleanup()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"\n❌ 对话处理错误: {e}")
                import traceback
                traceback.print_exc()
                error_text = "抱歉，我刚才有点走神了，我们继续聊吧。"
                tts_input_q.put(TextData(text=error_text, is_finish=True))
                continue
    
    # 启动线程
    asr_thread = threading.Thread(target=process_asr_output, name="ASR处理")
    conv_thread = threading.Thread(target=process_conversation, name="对话处理")
    
    asr_thread.daemon = True
    conv_thread.daemon = True
    
    asr_thread.start()
    conv_thread.start()
    
    # 等待线程结束
    try:
        while is_running:
            time.sleep(0.1)
    finally:
        asr_thread.join(timeout=1)
        conv_thread.join(timeout=1)

# ===================== TTS播放 =====================
def tts_to_play(tts_output_q: queue.Queue, audio_driver):
    """TTS合成结果 → 音频播放"""
    audio_chunk_count = 0
    
    while is_running:
        try:
            audio_data: AudioData = tts_output_q.get(timeout=0.1)
        except queue.Empty:
            continue
        
        if audio_data.pcm_data == b"":
            audio_driver.push_audio_for_play(audio_data)
            audio_chunk_count = 0
            continue
        
        audio_chunk_count += 1
        chunk_size = len(audio_data.pcm_data)
        
        if hasattr(audio_data, 'sample_rate') and audio_data.sample_rate > 0:
            bytes_per_sample = audio_data.bit_depth // 8 if hasattr(audio_data, 'bit_depth') else 2
            channels = audio_data.channels if hasattr(audio_data, 'channels') else 1
            samples = chunk_size / (bytes_per_sample * channels)
            duration_ms = (samples / audio_data.sample_rate) * 1000
            duration_str = f", 时长≈{duration_ms:.1f}ms"
        else:
            duration_str = ""
        
        print(f"🎵 音频分片 #{audio_chunk_count}, 大小: {chunk_size}字节{duration_str}")
        
        audio_driver.push_audio_for_play(audio_data)

# ===================== 按键控制 =====================
def key_control(audio_driver):
    """按键控制线程"""
    global is_recording, is_running, asr_input_q
    
    print("="*50)
    print("🎙️  流式语音交互系统")
    print("→ 按【空格键】：开始/停止语音输入")
    print("→ 按【ESC键】：退出程序")
    print("="*50)
    
    last_space_press = 0
    debounce_time = 0.5
    
    while is_running:
        try:
            current_time = time.time()
            
            # 空格键切换录音
            if keyboard.is_pressed('space') and (current_time - last_space_press) > debounce_time:
                last_space_press = current_time
                is_recording = not is_recording
                if is_recording:
                    print("\n▶️  开始录音...")
                    audio_driver.start_record(chunk_duration=0.6)
                else:
                    print("\n⏹️  停止录音...")
                    audio_driver.stop_record()
                    if asr_input_q is not None:
                        asr_input_q.put(AudioData(pcm_data=b"", sample_rate=16000, channels=1, is_finish=True))
            
            # ESC键退出
            if keyboard.is_pressed('esc'):
                print("\n🛑 退出程序...")
                is_running = False
                is_recording = False
                break
            
            time.sleep(0.05)
            
        except Exception as e:
            print(f"⌨️ 按键控制异常: {e}")
            break
    
    print("⌨️ 按键控制线程退出")

# ===================== 资源清理 =====================
def cleanup():
    """清理全局状态和资源"""
    global is_running, is_recording, asr_input_q
    is_running = False
    is_recording = False
    asr_input_q = None
    print("✅ 控制模块资源已清理")

# ===================== 记忆保存/加载 =====================
def save_memory(memory_system):
    """保存记忆到文件"""
    try:
        import json
        memory_data = {
            "long_term_memory": memory_system.long_term_memory,
            "user_profile": memory_system.user_profile,
            "save_time": time.time()
        }
        
        with open("memory_backup.json", "w", encoding="utf-8") as f:
            json.dump(memory_data, f, ensure_ascii=False, indent=2)
        print("✅ 记忆已保存")
    except Exception as e:
        print(f"❌ 记忆保存失败: {e}")

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
            print("✅ 记忆已加载")
            return memory_system
    except Exception as e:
        print(f"❌ 记忆加载失败: {e}")
    
    return MemorySystem()
def text_to_llm(text_input: str, tts_input_q: queue.Queue):
    """
    文本模式下的LLM-TTS异步流水线
    """
    memory_system = MemorySystem()
    
    # 创建智能分句器
    sentence_splitter = SmartSentenceSplitter(min_chunk_length=3, max_chunk_length=40)
    
    print(f"\n👤 用户输入: {text_input}")
    print("=" * 50)
    
    print(f"🤖 {name}: ", end="", flush=True)
    
    # 检查记忆关键词
    force_memory = any(keyword in text_input for keyword in ['之前', '刚才', '记得', '说过', '告诉过'])
    if force_memory:
        print("🧠 检测到记忆关键词，强制使用记忆...")
    
    # 开始异步流式生成
    start_time = time.time()
    tts_chunks_sent = 0
    full_response = ""
    
    try:
        # 使用异步流式生成器
        for chunk, is_final, final_response in create_async_stream_generator(
            text_input,
            memory_system=memory_system,
            temperature=0.2 if force_memory else 0.8
        ):
            if chunk:
                # 打印chunk
                print(chunk, end="", flush=True)
                full_response += chunk
                
                # 添加到分句器
                sentences = sentence_splitter.add_text(chunk)
                
                # 发送完整的句子到TTS
                for sentence in sentences:
                    if sentence.strip():
                        tts_chunks_sent += 1
                        # print(f"\n📤 发送TTS分片#{tts_chunks_sent}: {sentence}")
                        tts_input_q.put(TextData(text=sentence, is_finish=False))
            
            if is_final:
                # 发送剩余的文本
                remaining = sentence_splitter.flush()
                if remaining.strip():
                    tts_chunks_sent += 1
                    # print(f"\n📤 发送最终TTS分片#{tts_chunks_sent}: {remaining}")
                    tts_input_q.put(TextData(text=remaining, is_finish=False))
                
                # 发送结束标记
                tts_input_q.put(TextData(text="", is_finish=True))
                
                # 更新记忆
                memory_system.add_conversation(text_input, final_response)
                
                end_time = time.time()
                print(f"\n⏱️  响应时间: {end_time - start_time:.2f}秒")
                print(f"📤 共发送{tts_chunks_sent}个TTS分片")
                break
        
        print(f"\n{'=' * 50}")
        
        return full_response
        
    except Exception as e:
        print(f"\n❌ LLM处理错误: {e}")
        import traceback
        traceback.print_exc()
        error_text = "抱歉，我刚才有点走神了，我们继续聊吧。"
        tts_input_q.put(TextData(text=error_text, is_finish=True))
        return ""