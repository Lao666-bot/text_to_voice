"""
句子处理器：将ASR的片段累积成完整句子 - 修复版
"""

import queue
import time
from base_interface import TextData

class SentenceProcessor:
    """句子处理器：累积ASR片段，形成完整句子"""
    
    def __init__(self, min_length=3, max_silence=1.5):
        self.min_length = min_length
        self.max_silence = max_silence
        self.buffer = ""
        self.last_update = time.time()
        self.sentence_endings = ['。', '！', '？', '；', '.', '!', '?', ';']
    
    def process(self, text_data: TextData, output_queue: queue.Queue):
        """处理ASR文本，累积成完整句子后输出"""
        # 处理结束标记
        if text_data.is_finish and not text_data.text:
            if self.buffer:
                self._output_sentence(self.buffer, output_queue, True)
                self.buffer = ""
            output_queue.put(TextData(text="", is_finish=True))
            return
        
        text = text_data.text.strip()
        if not text:
            return
        
        # 清理多余的句号
        text = self._clean_punctuation(text)
        
        # 更新缓存
        self.buffer += text
        self.last_update = time.time()
        
        # 检查句子完整性
        if self._is_complete_sentence():
            self._output_sentence(self.buffer, output_queue, False)
            self.buffer = ""
        
        # 检查静音超时
        elif time.time() - self.last_update > self.max_silence and len(self.buffer) >= self.min_length:
            self._output_sentence(self.buffer, output_queue, True)
            self.buffer = ""
    
    def _clean_punctuation(self, text: str) -> str:
        """清理多余的标点符号"""
        # 移除句子中间的句号（保留其他标点）
        import re
        # 移除单独出现的句号（可能由ASR错误添加）
        text = re.sub(r'(?<!\w)\.(?!\w)', '', text)  # 移除单独的英文句号
        text = re.sub(r'(?<!\w)。(?!\w)', '', text)  # 移除单独的中文句号
        
        # 移除连续的句号
        text = re.sub(r'\.{2,}', '', text)
        text = re.sub(r'。{2,}', '', text)
        
        return text
    
    def _is_complete_sentence(self):
        """判断缓存是否构成完整句子"""
        if not self.buffer:
            return False
        
        # 以句子结束标点结尾
        if self.buffer[-1] in self.sentence_endings:
            return True
        
        # 长度足够且包含疑问词
        if len(self.buffer) > 8:
            question_words = ['吗', '呢', '吧', '啊', '什么', '为什么', '怎么', '如何', '谁', '哪']
            if any(word in self.buffer for word in question_words):
                return True
        
        return False
    
    def _output_sentence(self, sentence: str, output_queue: queue.Queue, is_timeout: bool):
        """输出完整句子到队列"""
        if not sentence:
            return
        
        # 清理句子：合并连续空格，去除首尾空白
        clean_sentence = ' '.join(sentence.split()).strip()
        
        if clean_sentence:
            reason = "超时" if is_timeout else "完整"
            ##print(f"📦 输出{reason}句子: {clean_sentence}")
            output_queue.put(TextData(text=clean_sentence, is_finish=True))
    
    def reset(self):
        """重置处理器状态"""
        self.buffer = ""
        self.last_update = time.time()


# 简单的句子处理器工厂函数
def create_sentence_processor(config=None):
    """创建句子处理器"""
    if config is None:
        config = {}
    
    min_length = config.get('min_length', 3)
    max_silence = config.get('max_silence', 1.5)
    
    return SentenceProcessor(min_length=min_length, max_silence=max_silence)