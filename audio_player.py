# audio_player.py（修复后完整代码）
import pyaudio
import numpy as np
import queue
import threading
from base_interface import AudioData


class AudioDriver:
    """音频驱动类：整合实时音频采集（麦克风）和播放功能，直接透传音频格式播放"""
    
    def __init__(self):
        # 初始化pyaudio核心实例
        self.p = pyaudio.PyAudio()
        # 播放模块状态
        self.is_playing = False
        self.play_thread = None
        self.audio_play_queue = queue.Queue()  # 播放队列
        # 采集模块状态
        self.is_recording = False
        self.record_thread = None
        self.audio_record_queue = queue.Queue()  # 采集队列
        # 【仅采集侧固定参数】播放侧完全透传TTS的音频格式
        self.sample_rate = 16000    # 仅用于采集
        self.channels = 1           # 仅用于采集
        self.format = pyaudio.paInt16  # 仅用于采集
        self.chunk_duration = 0.6
        self.chunk_samples = int(self.sample_rate * self.chunk_duration)
        # 新增：播放流缓存（避免重复创建/销毁）
        self.play_stream = None
        self.last_play_format = None
        self.last_play_rate = None
        self.last_play_channels = None

    # ===================== 音频播放相关方法（核心修改：常驻播放线程） =====================
    def _play_worker(self):
        """播放线程工作函数：常驻运行，仅处理结束信号不退出，透传TTS格式"""
        while self.is_playing:
            try:
                # 从播放队列取音频分片（超时0.1秒避免卡死）
                audio_data: AudioData = self.audio_play_queue.get(timeout=0.1)
                
                # 结束信号：仅清空当前播放流，不退出线程
                if audio_data is None or audio_data.pcm_data == b"":
                    # 仅关闭流但不退出线程，下次播放重新创建
                    if self.play_stream is not None:
                        self.play_stream.stop_stream()
                        self.play_stream.close()
                        self.play_stream = None
                        self.last_play_format = None
                        self.last_play_rate = None
                        self.last_play_channels = None
                    continue

                # 提取TTS返回的音频格式（优先使用AudioData自带的参数）
                current_format = self._get_pyaudio_format(audio_data)
                current_rate = audio_data.sample_rate
                current_channels = audio_data.channels

                # 格式变化/流未创建时重新打开播放流（适配TTS的任意格式）
                if (self.play_stream is None or 
                    current_format != self.last_play_format or 
                    current_rate != self.last_play_rate or 
                    current_channels != self.last_play_channels):
                    # 关闭旧的播放流（如果存在）
                    if self.play_stream is not None:
                        self.play_stream.stop_stream()
                        self.play_stream.close()
                    
                    # 打开新的播放流（使用TTS的音频格式）
                    self.play_stream = self.p.open(
                        format=current_format,
                        channels=current_channels,
                        rate=current_rate,
                        output=True,
                        frames_per_buffer=1024
                    )
                    # 更新格式缓存
                    self.last_play_format = current_format
                    self.last_play_rate = current_rate
                    self.last_play_channels = current_channels
                    # audio_player.py 修改 _play_worker 函数中的日志部分
                    print(f"🔄 适配TTS音频格式：采样率={current_rate}Hz, 声道={current_channels}, "
                        f"位深={self._get_bit_depth(current_format)}bit, "
                        f"数据大小={len(audio_data.pcm_data) if audio_data.pcm_data else 0}字节")
                    
                # 直接播放TTS生成的原始PCM数据（无任何转换）
                if self.play_stream is not None and audio_data.pcm_data:
                    self.play_stream.write(audio_data.pcm_data)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ 音频播放错误：{str(e)}")
                # 出错时重置播放流，不退出线程
                if self.play_stream is not None:
                    self.play_stream.stop_stream()
                    self.play_stream.close()
                    self.play_stream = None
                    self.last_play_format = None
                    self.last_play_rate = None
                    self.last_play_channels = None
                continue

        # 线程退出时最终释放播放流
        if self.play_stream is not None:
            self.play_stream.stop_stream()
            self.play_stream.close()
            self.play_stream = None

    def _get_pyaudio_format(self, audio_data: AudioData) -> int:
        """根据AudioData推导pyaudio格式（默认16bit）"""
        # 优先从AudioData获取位深，无则默认16bit
        bit_depth = getattr(audio_data, "bit_depth", 16)
        if bit_depth == 8:
            return pyaudio.paInt8
        elif bit_depth == 16:
            return pyaudio.paInt16
        elif bit_depth == 24:
            return pyaudio.paInt24
        elif bit_depth == 32:
            return pyaudio.paInt32
        elif bit_depth == 32 and getattr(audio_data, "is_float", False):
            return pyaudio.paFloat32
        else:
            # 默认返回16bit（兼容大部分TTS）
            return pyaudio.paInt16

    def _get_bit_depth(self, pyaudio_format: int) -> int:
        """反向推导位深（日志用）"""
        format_map = {
            pyaudio.paInt8: 8,
            pyaudio.paInt16: 16,
            pyaudio.paInt24: 24,
            pyaudio.paInt32: 32,
            pyaudio.paFloat32: 32
        }
        return format_map.get(pyaudio_format, 16)

    def start_play(self):
        """启动实时音频播放线程（常驻）"""
        if not self.is_playing:
            self.is_playing = True
            self.play_thread = threading.Thread(target=self._play_worker)
            self.play_thread.daemon = True  # 主线程退出时自动终止
            self.play_thread.start()
            print("✅ 音频播放线程已启动（透传TTS原始格式，常驻运行）")

    def stop_play(self):
        """停止音频播放并释放资源"""
        if self.is_playing:
            self.is_playing = False
            # 发送结束信号
            self.audio_play_queue.put(AudioData(pcm_data=b""))
            # 等待线程结束
            if self.play_thread:
                self.play_thread.join(timeout=2)
            print("✅ 音频播放线程已停止")

    def push_audio_for_play(self, audio_data: AudioData):
        """推送音频数据到播放队列（供TTS等模块调用）"""
        if self.is_playing:
            self.audio_play_queue.put(audio_data)

    # ===================== 音频采集相关方法（保持不变） =====================
    def _record_worker(self):
        """采集线程工作函数：循环采集麦克风数据并写入队列"""
        # 打开采集流（采集侧仍固定16kHz/单声道/16bit）
        record_stream = self.p.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_samples
        )

        while self.is_recording:
            try:
                # 读取麦克风PCM数据（忽略溢出异常）
                pcm_data = record_stream.read(self.chunk_samples, exception_on_overflow=False)
                # 封装为AudioData格式存入采集队列
                self.audio_record_queue.put(AudioData(pcm_data=pcm_data))
            except Exception as e:
                print(f"❌ 音频采集错误：{str(e)}")
                break

        # 释放采集流资源
        record_stream.stop_stream()
        record_stream.close()
        # 发送采集结束标记
        self.audio_record_queue.put(AudioData(pcm_data=b"", sample_rate=16000, channels=1, is_finish=True))

    def start_record(self, chunk_duration: float = None):
        """
        启动麦克风实时采集
        :param chunk_duration: 采集分片时长（秒），覆盖默认值
        """
        if not self.is_recording:
            # 覆盖分片时长（如果传入有效值）
            if chunk_duration and chunk_duration > 0:
                self.chunk_duration = chunk_duration
                self.chunk_samples = int(self.sample_rate * self.chunk_duration)
            # 启动采集线程
            self.is_recording = True
            self.record_thread = threading.Thread(target=self._record_worker)
            self.record_thread.daemon = True
            self.record_thread.start()
            print(f"✅ 音频采集线程已启动（分片时长：{self.chunk_duration}秒）")

    def stop_record(self):
        """停止麦克风采集并释放资源"""
        if self.is_recording:
            self.is_recording = False
            # 等待采集线程结束
            if self.record_thread:
                self.record_thread.join(timeout=2)
            print("✅ 音频采集线程已停止")

    def get_record_queue(self):
        """获取采集队列（供ASR模块读取音频数据）"""
        return self.audio_record_queue

    def get_play_queue(self):
        """获取播放队列（供外部模块监控播放状态）"""
        return self.audio_play_queue

    # ===================== 通用资源管理 =====================
    def release(self):
        """释放所有音频资源（析构时调用）"""
        # 停止所有线程
        self.stop_play()
        self.stop_record()
        # 终止pyaudio实例（增加空值判断）
        if self.p is not None:
            try:
                self.p.terminate()
            except:
                pass
        self.p = None  # 置空避免重复释放
        print("✅ 音频驱动所有资源已释放")


# ===================== 测试代码（验证多次音频输出） =====================
if __name__ == "__main__":
    # 测试多次音频播放（核心验证逻辑）
    audio_driver = AudioDriver()
    audio_driver.start_play()

    # 模拟多次TTS音频推送
    def simulate_multiple_tts():
        # 模拟第1次音频输出
        print("\n📢 第1次音频输出...")
        test_audio1 = AudioData(
            pcm_data=np.array([100, 200, 300], dtype=np.int16).tobytes(),
            sample_rate=16000,
            channels=1
        )
        audio_driver.push_audio_for_play(test_audio1)
        # 发送结束标记（测试流不退出）
        audio_driver.push_audio_for_play(AudioData(pcm_data=b""))
        time.sleep(1)

        # 模拟第2次音频输出
        print("\n📢 第2次音频输出...")
        test_audio2 = AudioData(
            pcm_data=np.array([400, 500, 600], dtype=np.int16).tobytes(),
            sample_rate=16000,
            channels=1
        )
        audio_driver.push_audio_for_play(test_audio2)
        audio_driver.push_audio_for_play(AudioData(pcm_data=b""))
        time.sleep(1)

        # 模拟第3次音频输出
        print("\n📢 第3次音频输出...")
        test_audio3 = AudioData(
            pcm_data=np.array([700, 800, 900], dtype=np.int16).tobytes(),
            sample_rate=16000,
            channels=1
        )
        audio_driver.push_audio_for_play(test_audio3)
        audio_driver.push_audio_for_play(AudioData(pcm_data=b""))

    import time
    simulate_multiple_tts()

    # 等待播放完成
    time.sleep(2)
    # 释放资源
    audio_driver.release()