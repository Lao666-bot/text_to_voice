# tts_driver.py（移除懒加载版本）
import os
import asyncio
import queue
import threading
import abc
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import time
import wave

os.environ["GENIE_DATA_DIR"] = r"C:\Users\k\Agent\Genie-TTS\GenieData"

# ===================== 1. 导入流式接口规范 =====================
@dataclass
class AudioData:
    pcm_data: bytes
    sample_rate: int = 16000
    channels: int = 1
    is_finish: bool = False  # 标记是否是最后一个音频分片
    bit_depth: int = 16      # 位深

@dataclass
class TextData:
    text: str
    is_finish: bool = True

ChatHistory = List[Dict[str, str]]

class BaseModule(abc.ABC):
    @abc.abstractmethod
    def process(self, input_data) -> Any:
        pass

    @abc.abstractmethod
    def stream_process(self, input_queue: queue.Queue, output_queue: queue.Queue):
        pass

# ===================== 2. 导入 Genie TTS 核心函数 =====================
from genie_tts import (
    load_character,
    load_predefined_character,
    set_reference_audio,
    tts,
    tts_async,  # 真正的流式接口
    unload_character,
    clear_reference_audio_cache,
    stop,
    wait_for_playback_done
)

# ===================== 3. 配置项（根据实际情况修改） =====================
LOCAL_MODEL_DIR = r"C:\Users\k\Agent\Genie-TTS\CharacterModels\v2ProPlus\feibi\tts_models"
LOCAL_CHAR_NAME = "feibi"
LOCAL_CHAR_LANG = "Chinese"
REFERENCE_AUDIO_PATH = r"C:\Users\k\Agent\Genie-TTS\CharacterModels\v2ProPlus\feibi\prompt_wav\zh_vo_Main_Linaxita_2_1_10_26.wav"
REFERENCE_AUDIO_TEXT = "在此之前,请您务必继续享受雨季拉古纳的时光"
SAVE_DIR = "./tts_output"

# ===================== 4. Genie TTS 流式模块实现（移除懒加载） =====================
class GenieTTSModule(BaseModule):
    def __init__(self):
        """初始化时立即加载模型（移除懒加载）"""
        print("🔄 TTS模块初始化中...")
        
        try:
            # 1. 加载TTS模型
            print(f"🔄 加载TTS模型: {LOCAL_CHAR_NAME}")
            load_character(
                character_name=LOCAL_CHAR_NAME,
                onnx_model_dir=LOCAL_MODEL_DIR,
                language=LOCAL_CHAR_LANG
            )
            print(f"✅ TTS模型 {LOCAL_CHAR_NAME} 加载成功")
            
            # 2. 设置参考音频
            print(f"🔄 设置参考音频: {REFERENCE_AUDIO_PATH}")
            set_reference_audio(
                character_name=LOCAL_CHAR_NAME,
                audio_path=REFERENCE_AUDIO_PATH,
                audio_text=REFERENCE_AUDIO_TEXT,
                language=LOCAL_CHAR_LANG
            )
            print(f"✅ 参考音频设置成功")
            
            # 3. 检测音频格式
            self._detect_audio_format()
            
            # 确保输出目录存在
            os.makedirs(SAVE_DIR, exist_ok=True)
            
            print("✅ TTS模块初始化完成")
            
        except Exception as e:
            print(f"❌ TTS模块初始化失败: {e}")
            raise

    def _detect_audio_format(self):
        """检测TTS音频格式"""
        try:
            # 生成一个简短的测试音频文件
            test_text = "测试音频格式"
            test_path = os.path.join(SAVE_DIR, "format_test.wav")
            
            print(f"🔄 检测音频格式，生成测试音频...")
            tts(
                character_name=LOCAL_CHAR_NAME,
                text=test_text,
                play=False,
                split_sentence=True,
                save_path=test_path
            )
            
            # 分析WAV文件格式
            with wave.open(test_path, 'rb') as wf:
                self.sample_rate = wf.getframerate()
                self.channels = wf.getnchannels()
                self.sample_width = wf.getsampwidth()
                self.bit_depth = self.sample_width * 8
                
                print(f"📊 TTS音频格式检测结果：")
                print(f"   采样率={self.sample_rate}Hz")
                print(f"   声道={self.channels}")
                print(f"   位深={self.bit_depth}bit")
                print(f"   样本宽度={self.sample_width}字节")
            
            # 清理测试文件
            try:
                os.remove(test_path)
            except:
                pass
                
        except Exception as e:
            print(f"❌ 音频格式检测失败: {e}")
            # 使用默认值
            print("⚠️  使用默认音频格式: 16000Hz/16bit/单声道")
            self.sample_rate = 16000
            self.channels = 1
            self.bit_depth = 16
            self.sample_width = 2

    def process(self, input_data: TextData) -> AudioData:
        """
        批量处理（非流式）：输入TextData，输出AudioData
        """
        # 1. 生成音频文件
        timestamp = int(time.time())
        save_path = os.path.join(SAVE_DIR, f"{LOCAL_CHAR_NAME}_{timestamp}.wav")
        
        ##print(f"🔄 批量处理TTS: {input_data.text[:50]}...")
        tts(
            character_name=LOCAL_CHAR_NAME,
            text=input_data.text,
            play=False,
            split_sentence=True,
            save_path=save_path
        )
        
        # 2. 读取音频文件为PCM数据
        with open(save_path, "rb") as f:
            pcm_data = f.read()
        
        ##print(f"✅ 批量TTS完成，音频大小: {len(pcm_data)} 字节")
        
        # 3. 返回AudioData格式
        return AudioData(
            pcm_data=pcm_data,
            sample_rate=self.sample_rate,
            channels=self.channels,
            bit_depth=self.bit_depth,
            is_finish=True
        )

    def stream_process(self, input_queue: queue.Queue, output_queue: queue.Queue):
        """实时流式处理：每收到一个句子就立即合成（同步版本）"""
        print("🔄 启动实时TTS流式处理...")
        
        sentence_count = 0
        
        while True:
            try:
                # 获取文本分片（增加超时时间）
                try:
                    text_data = input_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # 如果是结束标记
                if text_data.is_finish and not text_data.text:
                    output_queue.put(AudioData(
                        pcm_data=b"",
                        sample_rate=self.sample_rate,
                        channels=self.channels,
                        bit_depth=self.bit_depth,
                        is_finish=True
                    ))
                    ##print(f"✅ TTS流式处理完成，共合成{sentence_count}个句子")
                    break
                
                # 处理当前文本
                text = text_data.text.strip()
                if not text:
                    continue
                
                sentence_count += 1
                ##print(f"🎵 TTS开始合成句子 #{sentence_count}: {text[:50]}...")
                
                # 使用同步方法合成当前句子
                start_time = time.time()
                
                try:
                    # 为每个句子生成临时音频文件
                    timestamp = int(time.time())
                    save_path = os.path.join(SAVE_DIR, f"sentence_{timestamp}_{sentence_count}.wav")
                    
                    # 合成单个句子
                    tts(
                        character_name=LOCAL_CHAR_NAME,
                        text=text,
                        play=False,
                        split_sentence=False,  # 已经是完整句子，不需要再分割
                        save_path=save_path
                    )
                    
                    # 读取音频数据
                    with open(save_path, "rb") as f:
                        pcm_data = f.read()
                        
                    pcm_data = self._process_audio_start(pcm_data)
                    
                    elapsed = time.time() - start_time
                    
                    # 发送音频数据
                    output_queue.put(AudioData(
                        pcm_data=pcm_data,
                        sample_rate=self.sample_rate,
                        channels=self.channels,
                        bit_depth=self.bit_depth,
                        is_finish=False
                    ))
                    
                    ##print(f"✅ TTS句子 #{sentence_count} 合成完成，大小: {len(pcm_data)} 字节，耗时: {elapsed:.2f}秒")
                    
                    # 清理临时文件
                    try:
                        os.remove(save_path)
                    except:
                        pass
                    
                except Exception as e:
                    print(f"❌ TTS合成句子 #{sentence_count} 失败: {e}")
                    import traceback
                    traceback.print_exc()
                
            except Exception as e:
                print(f"❌ TTS流式处理错误: {e}")
                import traceback
                traceback.print_exc()
                break
    #======================去除开头的气泡音=====================
    def _process_audio_start(self, pcm_data: bytes) -> bytes:
        """
        处理音频开头的汽泡音
        移除开头的静音/噪声段
        """
        import numpy as np
        
        # 将字节转换为numpy数组
        dtype = np.int16 if self.bit_depth == 16 else np.int32
        samples = np.frombuffer(pcm_data, dtype=dtype)
        
        # 计算音频的RMS能量
        window_size = 100  # 10ms窗口（16000Hz采样率）
        num_windows = len(samples) // window_size
        
        # 寻找第一个非静音窗口
        start_index = 0
        silence_threshold = 500  # 调整这个阈值
        
        for i in range(min(10, num_windows)):  # 只检查前10个窗口（100ms）
            window = samples[i * window_size:(i + 1) * window_size]
            rms = np.sqrt(np.mean(window.astype(np.float64) ** 2))
            
            if rms > silence_threshold:
                # 找到语音开始，稍微提前一点（但不超过前一个窗口）
                start_index = max(0, (i - 1) * window_size)
                print(f"  检测到语音开始于第{i}个窗口，RMS={rms:.1f}")
                break
        
        # 如果没找到，尝试更宽松的条件
        if start_index == 0 and len(samples) > 2000:
            # 计算整个开头的RMS
            first_500 = samples[:2000]
            rms_500 = np.sqrt(np.mean(first_500.astype(np.float64) ** 2))
            
            if rms_500 < 100:  # 非常低的能量，可能是汽泡音
                # 直接跳过前50ms（800个样本，16kHz）
                start_index = min(800, len(samples) // 2)
                print(f"  低能量开头，跳过前{start_index}个样本")
        
        # 应用淡入效果，减少突变
        if start_index > 0:
            # 创建一个淡入窗口（20ms）
            fade_in_length = min(320, start_index)  # 320 samples = 20ms @ 16kHz
            
            # 复制原始音频
            processed_samples = samples[start_index:].copy()
            
            # 添加淡入效果
            if fade_in_length > 0 and len(processed_samples) > fade_in_length:
                # 创建淡入曲线（线性）
                fade_in = np.linspace(0, 1, fade_in_length)
                processed_samples[:fade_in_length] = (processed_samples[:fade_in_length] * fade_in).astype(dtype)
            
            # 转换回字节
            return processed_samples.tobytes()
        else:
            # 没有找到汽泡音，返回原始数据
            return pcm_data
    def __del__(self):
        """清理资源"""
        try:
            print("🧹 清理TTS模块资源...")
            unload_character(character_name=LOCAL_CHAR_NAME)
            clear_reference_audio_cache()
            print("✅ TTS模块资源已清理")
        except:
            pass


# ===================== 5. 测试代码 =====================
if __name__ == "__main__":
    # 测试TTS模块
    print("🧪 测试TTS模块...")
    
    try:
        # 创建TTS模块（立即加载模型）
        tts_module = GenieTTSModule()
        
        # 创建测试队列
        test_input_queue = queue.Queue()
        test_output_queue = queue.Queue()
        
        # 创建测试文本
        test_text = "你好，这是一个测试文本，用于验证TTS模块是否能正常工作。"
        print(f"📝 测试文本: {test_text}")
        
        # 将文本放入输入队列
        test_input_queue.put(TextData(text=test_text, is_finish=True))
        
        # 启动流式处理
        print("🔄 开始流式TTS测试...")
        tts_module.stream_process(test_input_queue, test_output_queue)
        
        # 检查输出
        chunk_count = 0
        while True:
            try:
                audio_data = test_output_queue.get(timeout=2.0)
                if audio_data.pcm_data:
                    chunk_count += 1
                    print(f"📊 收到音频分片 #{chunk_count}, 大小: {len(audio_data.pcm_data)} 字节")
                elif audio_data.is_finish:
                    print(f"✅ 测试完成，共收到 {chunk_count} 个音频分片")
                    break
            except queue.Empty:
                print("⏳ 等待音频分片...")
                break
        
        print("🎉 TTS模块测试完成")
        
    except Exception as e:
        print(f"❌ TTS测试失败: {e}")
        import traceback
        traceback.print_exc()