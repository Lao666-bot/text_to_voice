# audio_debug.py
import queue
import time
import threading
import wave
import os
from tts_driver import GenieTTSModule
from base_interface import TextData, AudioData

class AudioDebugger:
    """音频调试工具"""
    
    def __init__(self, tts_module):
        self.tts_module = tts_module
        self.audio_chunks = []
        self.audio_durations = []
        
    def test_tts_streaming(self, text, max_sentences=10):
        """测试TTS流式输出"""
        print(f"🔍 测试TTS流式输出: '{text[:50]}...'")
        
        # 创建队列
        input_queue = queue.Queue()
        output_queue = queue.Queue()
        
        # 将文本分割成句子
        sentences = self._split_into_sentences(text, max_sentences)
        
        print(f"📝 将文本分割为 {len(sentences)} 个句子:")
        for i, sentence in enumerate(sentences):
            print(f"  {i+1}. {sentence[:50]}...")
            input_queue.put(TextData(text=sentence, is_finish=(i == len(sentences)-1)))
        
        # 启动TTS处理线程
        def tts_worker():
            self.tts_module.stream_process(input_queue, output_queue)
        
        tts_thread = threading.Thread(target=tts_worker)
        tts_thread.start()
        
        # 收集音频数据
        chunk_count = 0
        start_time = time.time()
        
        while True:
            try:
                audio_data = output_queue.get(timeout=2.0)
                
                if audio_data.pcm_data == b"":
                    print(f"✅ 收到结束标记，共收到 {chunk_count} 个音频分片")
                    break
                
                chunk_count += 1
                self.audio_chunks.append(audio_data.pcm_data)
                
                # 计算时长
                if hasattr(audio_data, 'sample_rate') and audio_data.sample_rate > 0:
                    bytes_per_sample = audio_data.bit_depth // 8 if hasattr(audio_data, 'bit_depth') else 2
                    channels = audio_data.channels if hasattr(audio_data, 'channels') else 1
                    samples = len(audio_data.pcm_data) / (bytes_per_sample * channels)
                    duration_ms = (samples / audio_data.sample_rate) * 1000
                    self.audio_durations.append(duration_ms)
                    print(f"🎵 音频分片 #{chunk_count}: {len(audio_data.pcm_data)}字节, {duration_ms:.0f}ms")
                else:
                    print(f"🎵 音频分片 #{chunk_count}: {len(audio_data.pcm_data)}字节")
                
            except queue.Empty:
                print("⏳ 等待音频超时，可能已结束")
                break
        
        tts_thread.join(timeout=5)
        
        total_duration = sum(self.audio_durations) / 1000  # 转换为秒
        print(f"\n📊 统计:")
        print(f"  音频分片数量: {len(self.audio_chunks)}")
        print(f"  总音频时长: {total_duration:.2f}秒")
        print(f"  每个分片平均时长: {total_duration/len(self.audio_durations)*1000:.0f}ms" if self.audio_durations else "N/A")
        
        # 保存音频用于分析
        self._save_audio_for_analysis()
        
        return len(self.audio_chunks)
    
    def _split_into_sentences(self, text, max_sentences):
        """简单句子分割"""
        import re
        # 使用标点符号分割
        sentences = re.split(r'[。！？!?]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # 限制句子数量
        if len(sentences) > max_sentences:
            sentences = sentences[:max_sentences]
        
        return sentences
    
    def _save_audio_for_analysis(self):
        """保存音频用于分析"""
        if not self.audio_chunks:
            print("⚠️ 没有音频数据可保存")
            return
        
        os.makedirs("audio_debug", exist_ok=True)
        
        # 保存每个分片
        for i, chunk in enumerate(self.audio_chunks):
            filename = f"audio_debug/chunk_{i+1:03d}.wav"
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(1)  # 单声道
                wf.setsampwidth(2)  # 16bit = 2字节
                wf.setframerate(16000)  # 16kHz
                wf.writeframes(chunk)
            print(f"💾 保存分片 {i+1} 到 {filename}")
        
        # 合并所有分片
        if len(self.audio_chunks) > 1:
            combined_filename = "audio_debug/combined.wav"
            combined_data = b"".join(self.audio_chunks)
            with wave.open(combined_filename, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(combined_data)
            print(f"💾 合并音频保存到 {combined_filename}")

# 使用示例
if __name__ == "__main__":
    print("🔧 TTS音频调试工具")
    
    try:
        # 初始化TTS模块
        tts_module = GenieTTSModule()
        
        # 创建调试器
        debugger = AudioDebugger(tts_module)
        
        # 测试文本
        test_text = "这是一个测试文本，用于验证TTS流式输出是否正常。我们将检查音频分片是否完整，以及是否所有分片都能正确播放。如果发现问题，我们需要调试相关代码。"
        
        # 运行测试
        num_chunks = debugger.test_tts_streaming(test_text)
        
        print(f"\n🎉 调试完成，共生成 {num_chunks} 个音频分片")
        
    except Exception as e:
        print(f"❌ 调试失败: {e}")
        import traceback
        traceback.print_exc()