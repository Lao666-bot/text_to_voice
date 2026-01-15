# realtime_tts_processor.py
import queue
import threading
import time
import os
from base_interface import AudioData, TextData  # 从base_interface导入，保持一致

class RealtimeTTSProcessor:
    """实时TTS处理器：每收到一个句子立即合成"""
    
    def __init__(self, tts_module):
        self.tts_module = tts_module
        self.is_running = False
        self.thread = None
        
    def start_processing(self, input_queue: queue.Queue, output_queue: queue.Queue):
        """启动实时处理线程（修复结束逻辑）"""
        if self.is_running:
            print("⚠️  TTS处理器已经在运行")
            return
            
        self.is_running = True
        
        def process_loop():
            """处理循环"""
                                ##print("🎵 启动实时TTS处理器...")
            sentence_count = 0
            
            try:
                while self.is_running:
                    try:
                        # 获取文本
                        try:
                            text_data = input_queue.get(timeout=1.0)
                        except queue.Empty:
                            # 检查是否应该继续等待
                            continue
                        
                        # 检查结束标记
                        if text_data.is_finish and not text_data.text:
                            # 发送结束标记
                            output_queue.put(AudioData(
                                pcm_data=b"",
                                sample_rate=self.tts_module.sample_rate,
                                channels=self.tts_module.channels,
                                is_finish=True
                            ))
                                ##print(f"✅ TTS处理完成，共合成{sentence_count}个句子")
                            break
                        
                        text = text_data.text.strip()
                        if not text:
                            continue
                        
                        sentence_count += 1
                                ##print(f"🔊 TTS实时合成句子 #{sentence_count}: {text[:50]}...")
                        
                        # 记录开始时间
                        start_time = time.time()
                        
                        # 合成当前句子
                        try:
                            # 使用TTS模块的process方法
                            audio_data = self.tts_module.process(TextData(text=text, is_finish=True))
                            
                            elapsed = time.time() - start_time
                            
                            # 立即发送音频
                            output_queue.put(audio_data)
                            
                            ##print(f"✅ 句子 #{sentence_count} 合成完成，耗时: {elapsed:.2f}秒，大小: {len(audio_data.pcm_data)}字节")
                            
                        except Exception as e:
                            print(f"❌ TTS合成错误: {e}")
                            continue
                            
                    except Exception as e:
                        print(f"❌ TTS处理器错误: {e}")
                        import traceback
                        traceback.print_exc()
                        break
            
            finally:
                ##print("🛑 TTS处理器停止")
                self.is_running = False
        
        # 启动处理线程
        self.thread = threading.Thread(target=process_loop, daemon=True)
        self.thread.start()
        
    def stop(self):
        """停止处理器"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2)
            print("✅ TTS处理器已停止")