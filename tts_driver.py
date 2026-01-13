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
        
        print(f"🔄 批量处理TTS: {input_data.text[:50]}...")
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
        
        print(f"✅ 批量TTS完成，音频大小: {len(pcm_data)} 字节")
        
        # 3. 返回AudioData格式
        return AudioData(
            pcm_data=pcm_data,
            sample_rate=self.sample_rate,
            channels=self.channels,
            bit_depth=self.bit_depth,
            is_finish=True
        )

    def stream_process(self, input_queue: queue.Queue, output_queue: queue.Queue):
        """流式处理（实时生成音频）"""
        print("🔄 开始流式TTS处理...")
        
        # 收集所有文本分片
        full_text = ""
        text_data = None
        
        # 收集LLM输出的所有文本
        try:
            while True:
                try:
                    text_data = input_queue.get(timeout=1.0)
                    if text_data.text:
                        full_text += text_data.text
                        ##print(f"📝 收到文本分片: {text_data.text}")
                    if text_data.is_finish:
                        ##print("📝 收到文本结束标记")
                        break
                except queue.Empty:
                    ##print("⏳ 等待更多文本分片...")
                    continue
        except Exception as e:
            print(f"❌ 收集文本时出错: {e}")
            return
        
        if not full_text.strip():
            print("⚠️  文本为空，跳过TTS生成")
            output_queue.put(AudioData(
                pcm_data=b"",
                sample_rate=self.sample_rate,
                channels=self.channels,
                bit_depth=self.bit_depth,
                is_finish=True
            ))
            return
        
        print(f"🎵 开始TTS合成，文本长度: {len(full_text)} 字符")
        print(f"📄 文本内容: {full_text}")
        
        # 启动异步TTS生成
        async def generate_audio():
            try:
                chunk_count = 0
                total_bytes = 0
                
                print(f"🔄 调用tts_async生成音频...")
                async for audio_chunk in tts_async(
                    character_name=LOCAL_CHAR_NAME,
                    text=full_text,
                    play=False,
                    split_sentence=True,
                    save_path=None
                ):
                    if audio_chunk:
                        chunk_count += 1
                        total_bytes += len(audio_chunk)
                        
                        # 创建AudioData对象
                        audio_data = AudioData(
                            pcm_data=audio_chunk,
                            sample_rate=self.sample_rate,
                            channels=self.channels,
                            bit_depth=self.bit_depth,
                            is_finish=False
                        )
                        
                        # 推送到输出队列
                        output_queue.put(audio_data)
                        
                        print(f"🎵 生成音频分片 #{chunk_count}, 大小: {len(audio_chunk)} 字节")
                
                # 发送结束标记
                output_queue.put(AudioData(
                    pcm_data=b"",
                    sample_rate=self.sample_rate,
                    channels=self.channels,
                    bit_depth=self.bit_depth,
                    is_finish=True
                ))
                
                print(f"✅ TTS合成完成，共 {chunk_count} 个分片，总计 {total_bytes} 字节")
                
            except Exception as e:
                print(f"❌ TTS生成失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 运行异步TTS
        asyncio.run(generate_audio())
    
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