import abc
import threading
import queue
import os
import re
import time
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import soundfile
import numpy as np
from funasr import AutoModel

# 复用基础接口定义
@dataclass
class AudioData:
    pcm_data: bytes  
    sample_rate: int = 16000  
    channels: int = 1  
    is_finish: bool = False  # 补充is_finish字段，对齐其他模块

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

# FunASR流式识别驱动（集成标点恢复）
class FunASRStreamingASR(BaseModule):
    def __init__(self):
        # ========== 1. 初始化ASR模型（原有逻辑） ==========
        self.chunk_size = [0, 10, 5]  # 600ms chunk
        self.encoder_chunk_look_back = 4
        self.decoder_chunk_look_back = 1
        self.asr_model = AutoModel(
            model="paraformer-zh-streaming", 
            model_revision="v2.0.4",
            disable_update=True  # 禁用版本更新检查
        )
        self.chunk_stride = self.chunk_size[1] * 960  # 600ms stride
        
        # ========== 2. 初始化标点恢复模型（使用本地模型路径） ==========
        self.use_punc_model = False
        self.punc_model = None
        
        # 本地标点模型路径
        local_punc_model_path = r"C:\Users\k\.cache\modelscope\hub\iic\punc_ct-transformer_cn-en-common-vocab471067-large"
        
        # 优先尝试加载本地模型
        if os.path.exists(local_punc_model_path):
            try:
                print(f"🔄 尝试加载本地标点模型: {local_punc_model_path}")
                
                # 方法1：尝试通过本地路径加载
                try:
                    self.punc_model = AutoModel(
                        model=local_punc_model_path,
                        disable_update=True
                    )
                    self.use_punc_model = True
                    print(f"✅ 本地标点恢复模型加载成功: {local_punc_model_path}")
                except Exception as e1:
                    print(f"⚠️  通过本地路径加载失败，尝试其他方式: {e1}")
                    
                    # 方法2：尝试使用模型名称加载（可能已缓存）
                    self.punc_model = AutoModel(
                        model="iic/punc_ct-transformer_cn-en-common-vocab471067-large",
                        model_revision="v2.0.4",
                        disable_update=True
                    )
                    self.use_punc_model = True
                    print(f"✅ 通过模型名称加载标点模型成功")
                    
            except Exception as e:
                print(f"❌ 标点模型加载失败: {e}")
                self.use_punc_model = False
        else:
            print(f"⚠️  本地模型路径不存在: {local_punc_model_path}")
            
            # 备选方案：尝试加载公开模型
            print("🔄 尝试加载公开标点模型...")
            punc_model_candidates = [
                "ct-punc",
                "punc_ct-transformer_zh-cn-common-vocab272727",
                "punc_ct-transformer_cn",
            ]
            
            for model_name in punc_model_candidates:
                try:
                    print(f"🔄 尝试加载标点模型: {model_name}")
                    self.punc_model = AutoModel(
                        model=model_name,
                        model_revision="v1.0.2",
                        disable_update=True
                    )
                    self.use_punc_model = True
                    print(f"✅ 标点恢复模型加载成功: {model_name}")
                    break
                except Exception as e:
                    print(f"⚠️  模型 {model_name} 加载失败: {e}")
                    continue
        
        if not self.use_punc_model:
            print("⚠️  所有标点模型加载失败，降级为智能规则标点")
        else:
            print("✅ 标点恢复模块已就绪")
        
        # ========== 3. 流式缓存 ==========
        self.cache = {}  # ASR流式缓存
        self.punc_buffer = ""  # 标点恢复用文本缓存
        self.vad_active = False  # 简易VAD状态标记
        self.last_speech_time = time.time()
        self.silence_threshold = 1.0  # 静音阈值（秒）

    def _audio_data_to_numpy(self, audio_data: AudioData) -> np.ndarray:
        """格式转换：AudioData → numpy float32数组（适配FunASR）"""
        if audio_data.sample_rate != 16000:
            raise ValueError(f"仅支持16000Hz采样率，当前为{audio_data.sample_rate}Hz")
        if audio_data.channels != 1:
            raise ValueError(f"仅支持单声道，当前为{audio_data.channels}声道")
        
        speech = np.frombuffer(audio_data.pcm_data, dtype=np.int16)
        speech = speech.astype(np.float32) / 32767.0
        return speech

    def _simple_vad(self, speech_chunk: np.ndarray) -> bool:
        """简易VAD：通过音频能量判断是否有有效语音"""
        if len(speech_chunk) == 0:
            return False
        
        rms = np.sqrt(np.mean(np.square(speech_chunk)))
        return rms > 0.005  # 可根据环境调整阈值

    def _add_punctuation(self, text: str, is_final: bool = False) -> str:
        """
        核心：标点恢复逻辑（模型优先，规则降级）
        :param text: 无标点文本
        :param is_final: 是否是最后一个分片（决定是否清空缓存）
        :return: 带标点文本
        """
        if not text.strip():
            return ""
        
        # 步骤1：更新标点缓存（流式拼接）
        self.punc_buffer += text
        
        # 步骤2：仅在「有有效文本+（最后分片/缓存足够长）」时做标点恢复
        if len(self.punc_buffer) < 2 and not is_final:
            return self.punc_buffer  # 文本过短时暂不处理
        
        try:
            # 方案A：使用标点模型（优先）
            if self.use_punc_model and self.punc_model is not None:
                # 调用标点模型
                ##print(f"🔤 标点模型输入: {self.punc_buffer}")
                punc_result = self.punc_model.generate(input=self.punc_buffer)
                ##print(f"🔤 标点模型原始输出: {punc_result}")
                
                # 处理不同的返回格式
                punctuated_text = self._extract_text_from_punc_result(punc_result)
                    
                # 清理结果
                punctuated_text = punctuated_text.strip()
                
            # 方案B：智能规则标点（降级）
            else:
                punctuated_text = self._smart_rule_based_punc(self.punc_buffer)
        
        except Exception as e:
            print(f"⚠️  标点恢复失败，使用原文本：{e}")
            import traceback
            traceback.print_exc()
            punctuated_text = self.punc_buffer
        
        # 步骤3：最后分片时清空缓存，否则保留末尾字符（避免断句）
        if is_final:
            final_text = punctuated_text
            self.punc_buffer = ""  # 清空缓存
            ##print(f"✅ 最终标点结果: {final_text}")
        else:
            # 保留最后1-2个字符，避免断句（如"今天天气"→ 保留"天气"）
            final_text = punctuated_text[:-2] if len(punctuated_text) > 2 else ""
            self.punc_buffer = punctuated_text[-2:] if len(punctuated_text) > 2 else punctuated_text
            ##print(f"📝 临时标点结果: {final_text}")
        
        return final_text.strip()

    def _extract_text_from_punc_result(self, punc_result) -> str:
        """从标点模型结果中提取文本"""
        if punc_result is None:
            return self.punc_buffer
            
        # 处理不同的返回格式
        if isinstance(punc_result, list):
            if len(punc_result) > 0:
                if isinstance(punc_result[0], dict):
                    # 格式: [{'text': '带标点的文本'}]
                    return punc_result[0].get("text", self.punc_buffer)
                elif isinstance(punc_result[0], str):
                    # 格式: ['带标点的文本']
                    return punc_result[0]
                else:
                    # 尝试转换为字符串
                    return str(punc_result[0])
            else:
                return self.punc_buffer
        elif isinstance(punc_result, dict):
            # 格式: {'text': '带标点的文本'}
            return punc_result.get("text", self.punc_buffer)
        elif isinstance(punc_result, str):
            # 格式: '带标点的文本'
            return punc_result
        else:
            # 其他格式，尝试转换
            return str(punc_result) if punc_result else self.punc_buffer

    def _smart_rule_based_punc(self, text: str) -> str:
        """智能规则标点恢复（模型降级时使用）"""
        if not text.strip():
            return ""
        
        # 清理文本：去除多余的标点
        import re
        
        # 移除单独出现的标点
        text = re.sub(r'(?<!\w)[。！？；：，、\.!?;:,](?!\w)', '', text)
        
        # 移除连续重复的标点
        text = re.sub(r'[。]{2,}', '', text)
        text = re.sub(r'[！]{2,}', '', text)
        text = re.sub(r'[？]{2,}', '', text)
        text = re.sub(r'[\.]{2,}', '', text)
        text = re.sub(r'[!]{2,}', '', text)
        text = re.sub(r'[?]{2,}', '', text)
        
        # 判断句子类型并添加合适的标点
        if len(text) < 3:
            return text  # 短文本不加标点
        
        # 检查是否已经是完整句子（已有标点）
        if re.search(r'[。！？\.!?]$', text):
            return text
        
        # 疑问词列表
        question_words = ['吗', '呢', '吧', '啊', '什么', '为什么', '怎么', '如何', 
                         '谁', '哪', '哪里', '多少', '几', '怎样', '为何']
        
        # 感叹词列表
        exclamation_words = ['啊', '呀', '哇', '哦', '唉', '哈', '嘿', '喂']
        
        # 检查是否包含疑问词
        has_question = any(word in text for word in question_words)
        # 检查是否包含感叹词
        has_exclamation = any(word in text for word in exclamation_words)
        
        # 检查是否以疑问词结尾
        ends_with_question = text.endswith(tuple(question_words))
        
        if has_question or ends_with_question:
            # 疑问句加问号
            return text + '？'
        elif has_exclamation:
            # 感叹句加感叹号
            return text + '！'
        else:
            # 陈述句加句号（但只在句子较长时）
            if len(text) > 6:
                return text + '。'
            else:
                return text

    def _numpy_to_text_data(self, asr_result: str, is_finish: bool) -> TextData:
        """格式转换：识别结果 → TextData"""
        return TextData(
            text=asr_result,
            is_finish=is_finish
        )

    def process(self, input_data: AudioData) -> TextData:
        """批量处理：完整音频识别+标点恢复"""
        # 1. 音频格式转换
        speech = self._audio_data_to_numpy(input_data)
        total_chunk_num = int((len(speech)-1)/self.chunk_stride + 1)
        final_text = ""
        cache = {}

        # 2. 逐chunk识别
        for i in range(total_chunk_num):
            speech_chunk = speech[i*self.chunk_stride:(i+1)*self.chunk_stride]
            is_final = i == total_chunk_num - 1
            
            # 3. 流式ASR识别
            res = self.asr_model.generate(
                input=speech_chunk,
                cache=cache,
                is_final=is_final,
                chunk_size=self.chunk_size,
                encoder_chunk_look_back=self.encoder_chunk_look_back,
                decoder_chunk_look_back=self.decoder_chunk_look_back
            )
            
            # 4. 提取识别文本
            chunk_text = res[0]["text"] if res and len(res) > 0 else ""
            final_text += chunk_text

        # 5. 全局标点恢复
        final_text = self._add_punctuation(final_text, is_final=True)
        return self._numpy_to_text_data(final_text, is_finish=True)

    # 在 FunASRStreamingASR 类的 stream_process 方法中修改：
    def stream_process(self, input_queue: queue.Queue, output_queue: queue.Queue):
        """流式处理：音频分片识别+实时标点恢复"""
        # 重置所有缓存
        self.cache = {}
        self.punc_buffer = ""  # 存储未加标点的原始文本
        self.vad_active = False
        self.last_speech_time = time.time()
        
        # 新增：完整句子缓存（用于标点恢复）
        self.sentence_buffer = ""
        self.sentence_complete = False

        try:
            while True:
                # 1. 从队列获取音频分片
                try:
                    audio_chunk: AudioData = input_queue.get(timeout=1.0)
                except queue.Empty:
                    # 检查静音超时 - 简化逻辑
                    if (time.time() - self.last_speech_time > self.silence_threshold and 
                        self.sentence_buffer):
                        # 静音超时，处理缓存的句子
                        final_text = self._process_sentence(self.sentence_buffer, is_final=True)
                        if final_text:
                            output_queue.put(self._numpy_to_text_data(final_text, is_finish=True))
                        self.sentence_buffer = ""
                    continue
                
                # 结束标记
                if audio_chunk.pcm_data == b"" and audio_chunk.is_finish:
                    # 处理最后缓存的文本
                    if self.sentence_buffer:
                        final_text = self._process_sentence(self.sentence_buffer, is_final=True)
                        output_queue.put(self._numpy_to_text_data(final_text, is_finish=True))
                    # 推送结束标记
                    output_queue.put(self._numpy_to_text_data("", is_finish=True))
                    ##print("🔤 ASR处理完成")
                    break

                # 2. 格式转换
                speech_chunk = self._audio_data_to_numpy(audio_chunk)
                
                # 3. 简易VAD过滤静音
                is_speech = self._simple_vad(speech_chunk)
                current_time = time.time()
                
                if is_speech:
                    self.last_speech_time = current_time
                    self.vad_active = True
                    
                    # 4. 流式ASR识别
                    res = self.asr_model.generate(
                        input=speech_chunk,
                        cache=self.cache,
                        is_final=False,
                        chunk_size=self.chunk_size,
                        encoder_chunk_look_back=self.encoder_chunk_look_back,
                        decoder_chunk_look_back=self.decoder_chunk_look_back
                    )
                    
                    # 5. 提取识别文本
                    chunk_text = res[0]["text"] if res and len(res) > 0 else ""
                    
                    if chunk_text:
                        print(f"🔤 ASR识别: {chunk_text}")
                        
                        # 6. 累积到句子缓存
                        self.sentence_buffer += chunk_text
                        
                        # 7. 检查句子是否自然结束（中文常见结束词）
                        # 如果句子较长且有明显的结束词，可以提前处理
                        if len(self.sentence_buffer) >= 8:  # 句子较长时
                            # 检查是否有自然结束词
                            end_words = ['吗', '呢', '吧', '啊', '呀', '哦', '哈', '啦', '的', '了']
                            if any(self.sentence_buffer.endswith(word) for word in end_words):
                                # 提前处理句子
                                final_text = self._process_sentence(self.sentence_buffer, is_final=False)
                                if final_text:
                                    # 只输出已经完成的句子部分
                                    output_queue.put(self._numpy_to_text_data(final_text, is_finish=False))
                                    # 清空缓存，但保留最后几个字符以防断句
                                    self.sentence_buffer = self.sentence_buffer[-3:] if len(self.sentence_buffer) > 3 else ""
                        
                elif self.vad_active and not is_speech:
                    # VAD从激活变静音，处理完整句子
                    silence_duration = current_time - self.last_speech_time
                    if silence_duration > 0.5 and self.sentence_buffer:  # 0.5秒静音
                        # 处理缓存的句子
                        final_text = self._process_sentence(self.sentence_buffer, is_final=True)
                        if final_text:
                            output_queue.put(self._numpy_to_text_data(final_text, is_finish=True))
                        self.sentence_buffer = ""
                        self.vad_active = False

        except Exception as e:
            print(f"❌ ASR流式处理异常: {e}")
            import traceback
            traceback.print_exc()
            output_queue.put(self._numpy_to_text_data("", is_finish=True))

    def _process_sentence(self, text: str, is_final: bool = False) -> str:
        """处理句子：添加标点"""
        if not text.strip():
            return ""
        
        # 移除可能的重复文本（简单去重）
        # 如果文本以标点结尾，可能是上次残留的
        import re
        text = text.strip()
        
        # 处理重复文本（如"赢？赢？" -> "赢？"）
        if len(text) >= 4 and text[-2:] == text[-4:-2]:
            # 发现重复，移除后半部分
            half_len = len(text) // 2
            if text[:half_len] == text[half_len:]:
                text = text[:half_len]
        
        ##print(f"📝 处理句子: '{text}' (is_final: {is_final})")
        
        try:
            if self.use_punc_model and self.punc_model is not None:
                ##print(f"🔤 调用标点模型: '{text}'")
                punc_result = self.punc_model.generate(input=text)
                ##print(f"🔤 标点模型输出: {punc_result}")
                
                punctuated_text = self._extract_text_from_punc_result(punc_result)
                punctuated_text = punctuated_text.strip()
                
                # 清理标点：移除连续的标点
                punctuated_text = re.sub(r'([。！？])\1+', r'\1', punctuated_text)
                punctuated_text = re.sub(r'([,，])\1+', r'\1', punctuated_text)
                
                print(f"✅ 标点结果: '{punctuated_text}'")
                return punctuated_text
            else:
                # 使用规则标点
                return self._smart_rule_based_punc(text)
                
        except Exception as e:
            print(f"⚠️ 标点处理失败: {e}")
            return text  # 返回原文本