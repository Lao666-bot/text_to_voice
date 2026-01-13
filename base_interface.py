import abc  # 导入抽象基类模块：用于定义“必须实现的方法”
import threading  # 用于流式处理的多线程
import queue  # 用于流式处理的队列（数据传输通道）
from dataclasses import dataclass
from typing import List, Optional, Dict, Any  # 把Any加到已有的typing导入里

# 2. 定义音频数据的统一格式（所有模块交互音频时，都用这个类）
@dataclass  # 装饰器：自动生成初始化/打印等方法，简化代码
class AudioData:
    # 核心字段1：PCM裸数据（bytes类型）→ 音频的原始二进制数据，int16格式（16位采样）
    pcm_data: bytes  
    # 核心字段2：采样率 → 固定为16000Hz（FunASR原生支持，GPT-VITS可适配），默认值避免重复传参
    sample_rate: int = 16000  
    # 核心字段3：声道数 → 固定单声道（语音识别/合成都用单声道）
    channels: int = 1  
    is_finish: bool = False  # 新增：标记是否是最后一个音频分片

# 3. 定义文本数据的统一格式（LLM/ASR/TTS交互文本时用）
@dataclass
class TextData:
    # 核心字段1：纯文本内容（UTF-8编码，避免乱码）
    text: str  
    # 核心字段2：是否是最后一个分片（流式输出关键）→ 默认True（批量输出），流式时设为False
    is_finish: bool = True  

# 4. 定义对话上下文的统一格式（LLM多轮对话专用）
# 类型别名：把「List[Dict[str, str]]」简化为「ChatHistory」，代码更易读
ChatHistory = List[Dict[str, str]]  
# 格式要求：每个字典必须有"role"（角色：user/assistant）和"content"（内容）
# 示例：ChatHistory = [{"role":"user","content":"你好"}, {"role":"assistant","content":"你好呀"}]
# 1. 定义所有模块的抽象基类（相当于“接口规范”）
class BaseModule(abc.ABC):  # 继承abc.ABC表示这是抽象类，不能直接实例化
    # 2. 定义批量处理方法（非流式）：所有模块必须实现
    @abc.abstractmethod  # 装饰器：强制子类实现这个方法，否则报错
    def process(self, input_data) -> Any:
        """
        批量处理：一次性输入、一次性输出
        :param input_data: 输入数据（AudioData/TextData）
        :return: 输出数据（AudioData/TextData）
        """
        pass  # pass表示“占位”，子类必须替换为实际逻辑

    # 3. 定义流式处理方法（核心）：所有模块必须实现
    @abc.abstractmethod
    def stream_process(self, input_queue: queue.Queue, output_queue: queue.Queue):
        """
        流式处理：从输入队列取分片数据，处理后推到输出队列
        :param input_queue: 输入队列（放AudioData/TextData分片）
        :param output_queue: 输出队列（推处理后的分片）
        """
        pass# 1. 导入必要的工具：dataclass用于定义结构化数据，typing用于类型标注（新手可先记：让代码更易读/易维护）