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
import logging
os.environ["GENIE_DATA_DIR"] = r"C:\Users\k\Agent\Genie-TTS\GenieData"
#======================这是一个日志过滤器，用于过滤掉特定的警告======================
class GenieTTSFilter(logging.Filter):
    def filter(self, record):
        # 过滤掉包含 "Audio successfully saved" 的日志
        if "Audio successfully saved" in record.getMessage():
            return False
        return True
logging.getLogger().addFilter(GenieTTSFilter())
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
                        ##去除开头的气泡音
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
        专门处理TTS开头爆破音的函数
        爆破音特征：低频能量高、突然的能量爆发、持续时间短（<50ms）
        """
        import numpy as np
        
        # 将字节转换为numpy数组
        dtype = np.int16 if self.bit_depth == 16 else np.int32
        samples = np.frombuffer(pcm_data, dtype=dtype)
        
        if len(samples) < 1600:  # 小于100ms的音频不处理
            return pcm_data
        
        # 1. 爆破音专用检测算法
        def detect_plosive_noise(audio_data):
            """检测爆破音噪音"""
            # 分析前100ms（1600个样本）
            analysis_length = min(1600, len(audio_data))
            segment = audio_data[:analysis_length].astype(np.float32)
            
            # 计算短期能量（用于检测突发能量）
            window_size = 160  # 10ms窗口
            num_windows = analysis_length // window_size
            
            energies = []
            for i in range(num_windows):
                window = segment[i * window_size:(i + 1) * window_size]
                energy = np.sum(window ** 2) / window_size
                energies.append(energy)
            
            # 计算能量变化率（爆破音的特点是能量突然增加）
            energy_diffs = np.diff(energies)
            
            # 检测能量突然爆发的点
            sudden_increase_threshold = np.max(energies) * 0.3
            
            for i in range(1, len(energy_diffs)):
                if energy_diffs[i] > sudden_increase_threshold:
                    # 爆破音通常在前3个窗口内
                    if i * window_size < 480:  # 前30ms内
                        # 向前找更合适的起始点（可能在爆发的稍前位置）
                        return max(0, (i - 1) * window_size)
            
            return 0
        
        # 2. 低频爆破音检测（爆破音通常在低频）
        def detect_low_freq_plosive(audio_data):
            """检测低频爆破音"""
            try:
                from scipy import signal
                
                analysis_length = min(800, len(audio_data))
                segment = audio_data[:analysis_length].astype(np.float32)
                
                # 设计带通滤波器（50-200Hz，爆破音主要频率范围）
                lowcut = 50
                highcut = 200
                nyquist = self.sample_rate / 2
                
                # 巴特沃斯带通滤波器
                b, a = signal.butter(
                    4, 
                    [lowcut/nyquist, highcut/nyquist], 
                    btype='band'
                )
                
                # 应用滤波器
                filtered = signal.filtfilt(b, a, segment)
                
                # 计算滤波后的能量
                window_size = 80  # 5ms
                num_windows = analysis_length // window_size
                
                filtered_energies = []
                for i in range(num_windows):
                    window = filtered[i * window_size:(i + 1) * window_size]
                    energy = np.sum(window ** 2) / window_size
                    filtered_energies.append(energy)
                
                # 找到第一个低频能量峰值
                energy_threshold = np.percentile(filtered_energies, 70)
                
                for i, energy in enumerate(filtered_energies):
                    if energy > energy_threshold:
                        # 爆破音通常持续1-2个窗口（5-10ms）
                        return max(0, (i - 1) * window_size)
                        
            except ImportError:
                # scipy不可用时使用简化方法
                pass
            
            return 0
        
        # 3. 经验法则：根据TTS引擎特性直接切除
        def empirical_cut_for_tts():
            """根据经验直接切除固定长度"""
            # Genie TTS通常在开头有固定模式的噪音
            # 尝试切除前30-50ms（480-800个样本）
            
            # 先检查前100ms的能量分布
            first_100ms = min(1600, len(samples))
            
            # 分成4个25ms的窗口
            window_25ms = 400  # 16kHz * 0.025s
            windows = []
            
            for i in range(0, first_100ms, window_25ms):
                if i + window_25ms <= first_100ms:
                    window = samples[i:i+window_25ms]
                    rms = np.sqrt(np.mean(window.astype(np.float64) ** 2))
                    windows.append(rms)
            
            # 如果第一个窗口能量明显高于后面，很可能是噪音
            if len(windows) >= 2 and windows[0] > windows[1] * 1.5:
                return 400  # 切除前25ms
            
            # 默认切除30ms（480个样本）
            return 480
        
        # 4. 波形形状检测（爆破音的波形特征）
        def detect_by_waveform_shape(audio_data):
            """通过波形形状检测爆破音"""
            analysis_length = min(800, len(audio_data))
            segment = audio_data[:analysis_length]
            
            # 计算波形的一阶和二阶差分（检测突变）
            diff1 = np.diff(segment)
            diff2 = np.diff(diff1)
            
            # 寻找幅度突变点
            amplitude_threshold = np.percentile(np.abs(diff1), 90)
            
            for i in range(len(diff1) - 10):
                # 检查是否有一系列的突变
                if np.abs(diff1[i]) > amplitude_threshold:
                    # 检查后续几个点是否也有较大变化
                    subsequent = np.abs(diff1[i:i+10])
                    if np.mean(subsequent) > amplitude_threshold * 0.5:
                        return max(0, i - 20)  # 稍微提前一点
            
            return 0
        
        # 5. 综合多种检测方法
        def combined_detection():
            """综合使用多种检测方法"""
            detection_results = []
            
            # 方法1：能量突变检测
            pos1 = detect_plosive_noise(samples)
            if pos1 > 0:
                detection_results.append(pos1)
            
            # 方法2：低频检测（需要scipy）
            pos2 = detect_low_freq_plosive(samples)
            if pos2 > 0:
                detection_results.append(pos2)
            
            # 方法3：波形形状检测
            pos3 = detect_by_waveform_shape(samples)
            if pos3 > 0:
                detection_results.append(pos3)
            
            # 方法4：经验切除
            pos4 = empirical_cut_for_tts()
            detection_results.append(pos4)
            
            # 如果所有方法都认为有噪音，取中间值
            if detection_results:
                # 去掉最大最小值，取中间值
                sorted_results = sorted(detection_results)
                if len(sorted_results) >= 3:
                    # 取中位数
                    return sorted_results[len(sorted_results) // 2]
                else:
                    # 取平均值
                    return int(np.mean(sorted_results))
            
            return 480  # 默认切除30ms
        
        # 执行检测
        start_index = combined_detection()
        
        # 确保不会切除太多（不超过20%，且不超过200ms）
        max_cut = min(len(samples) // 5, 3200)  # 200ms或20%
        start_index = min(start_index, max_cut)
        
        # 应用切除
        if start_index > 0:
            # 添加更长的淡入效果来平滑过渡（50ms）
            fade_in_length = min(800, len(samples) - start_index)  # 50ms淡入
            
            # 复制切除后的音频
            processed_samples = samples[start_index:].copy()
            
            if fade_in_length > 0 and len(processed_samples) > fade_in_length:
                # 使用更平滑的淡入曲线（余弦曲线）
                fade_in = np.cos(np.linspace(np.pi/2, 0, fade_in_length))
                processed_samples[:fade_in_length] = (processed_samples[:fade_in_length] * fade_in).astype(dtype)
                
                print(f"  ✂️ 切除 {start_index} 样本 ({start_index/self.sample_rate*1000:.0f}ms)")
            
            # 确保切除后音频不会太短
            if len(processed_samples) > 1600:  # 至少100ms
                return processed_samples.tobytes()
        
        # 如果没有切除或切除后太短，返回原始数据
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