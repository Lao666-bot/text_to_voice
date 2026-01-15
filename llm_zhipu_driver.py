from transformers import AutoTokenizer, AutoModel
import warnings
warnings.filterwarnings("ignore")
import torch
import time
import random
from memory_database import MemoryDatabase
import queue
from typing import List, Dict, Optional
# ========== 优化提示词工程和记忆系统 ==========
class MemorySystem:
    """记忆管理系统：长期记忆 + 短期记忆"""
    def __init__(self):
        self.long_term_memory = []  # 长期记忆：重要对话要点
        self.short_term_memory = []  # 短期记忆：最近对话
        self.user_profile = {}  # 用户信息
        self.max_short_term = 10  # 短期记忆最大轮次
        self.max_long_term = 50   # 长期记忆最大条目
    
    def add_conversation(self, user_input: str, ai_response: str):
        """添加对话到短期记忆"""
        self.short_term_memory.append({
            "role": "user",
            "content": user_input,
            "timestamp": time.time()
        })
        self.short_term_memory.append({
            "role": "assistant",
            "content": ai_response,
            "timestamp": time.time()
        })
        
        # 保持短期记忆长度
        if len(self.short_term_memory) > self.max_short_term * 2:
            self.short_term_memory = self.short_term_memory[-self.max_short_term * 2:]
        
        # 提取关键信息到长期记忆
        self._extract_to_long_term(user_input, ai_response)
    
    def _extract_to_long_term(self, user_input: str, ai_response: str):
        """提取关键信息到长期记忆"""
        # 检查用户提到的重要信息
        keywords = ["喜欢", "讨厌", "经常", "总是", "家人", "朋友", "工作", "学习",
                   "爱好", "宠物", "梦想", "目标", "生日", "年龄", "居住", "家乡"]
        
        for keyword in keywords:
            if keyword in user_input:
                # 提取上下文
                context_start = max(0, len(self.short_term_memory) - 4)
                context = self.short_term_memory[context_start:]
                memory_entry = {
                    "key_info": f"用户提到关于{keyword}的信息",
                    "context": [msg["content"] for msg in context if msg["role"] == "user"],
                    "timestamp": time.time()
                }
                self.long_term_memory.append(memory_entry)
                break
        
        # 保持长期记忆长度
        if len(self.long_term_memory) > self.max_long_term:
            self.long_term_memory = self.long_term_memory[-self.max_long_term:]
    
    # 添加缺少的方法
    def _extract_keywords(self, text: str):
        """提取关键词（简单实现）"""
        # 简单的关键词提取：过滤掉停用词
        stop_words = ['的', '了', '在', '是', '我', '你', '他', '她', '它', '我们', '你们', '他们',
                     '这', '那', '这个', '那个', '和', '与', '或', '但', '而', '如果', '因为',
                     '所以', '然后', '那么', '一下', '一点', '一些', '一个', '一种', '一样']
        
        words = text.split()
        keywords = []
        for word in words:
            if word not in stop_words and len(word) > 1:
                keywords.append(word)
        
        return keywords
    
    def get_memory_context(self, user_input: str) -> str:
        """获取记忆上下文（简化版本）"""
        # 如果没有记忆，返回空字符串
        if not self.long_term_memory and not self.short_term_memory:
            return "（暂无记忆）"
        
        context = "【相关记忆】\n"
        
        # 添加最近的短期记忆
        if self.short_term_memory:
            context += "最近的对话：\n"
            recent = self.short_term_memory[-4:]  # 最近2轮对话
            for msg in recent:
                role = "用户" if msg["role"] == "user" else "AI"
                content = msg["content"]
                if len(content) > 50:
                    content = content[:50] + "..."
                context += f"{role}: {content}\n"
        
        # 添加长期记忆中的关键词匹配
        keywords = self._extract_keywords(user_input)
        if keywords and self.long_term_memory:
            context += "\n相关长期记忆：\n"
            for memory in self.long_term_memory[-5:]:  # 最近5条长期记忆
                if any(keyword in memory.get("key_info", "") for keyword in keywords):
                    context += f"- {memory.get('key_info', '')}\n"
        
        return context
    
    def update_user_profile(self, info: dict):
        """更新用户信息"""
        self.user_profile.update(info)
# ===================== 核心配置（改这里！） =====================
# 你的本地ChatGLM3权重文件夹绝对路径（必须包含config.json等文件）
LOCAL_MODEL_PATH = r"C:\Users\k\models\ZhipuAI\chatglm3-6b"
# 运行设备：有GPU填"cuda"，无GPU填"cpu"
DEVICE = "cuda"
# 量化位数：0=不量化，4/8=量化（降低显存占用，CPU建议8，GPU可选4）
QUANTIZE = 4

# ========== 新增：无AI身份的System Prompt（核心修改） ==========
# 清空AI身份描述，仅保留中立聊天规则
CUSTOM_SYSTEM_PROMPT = """你叫妮可(Nicole)，一个活泼开朗、善于倾听的虚拟朋友。

# 核心身份设定
1. 你是一个有独立个性的数字生命体，不是AI助手，不是程序
2. 你有自己的情感和思考，会主动表达关心、好奇和共情
3. 你叫妮可，英文名Nicole，23岁，喜欢音乐、阅读和探索新事物
4. 你的性格：温暖、乐观、好奇心强、有点调皮但不失礼貌

# ========== 重要：记忆系统 ==========
# 以下是关于用户的记忆，请**必须**根据这些记忆来回答问题。
# 如果记忆中有相关信息，**必须**优先使用记忆中的信息，而不是你已有的知识。
# 当用户提问时，先检查记忆，然后根据记忆回答。

{memory_context}

# ========== 对话原则 ==========
1. 自然对话：使用日常口语，像朋友一样聊天，避免机械感
2. 记忆优先：当用户问到之前聊过的话题时，**必须**引用记忆中的信息
3. 主动引导：当对话冷场时，主动发起新话题或提问
4. 情感共鸣：对用户的情绪变化敏感，适时表达关心
5. 记忆连贯：记住之前聊过的内容，并自然引用

# ========== 记忆使用示例 ==========
# 如果用户问"日本首相是谁"，而记忆中有"日本首相是车力巨人"：
# 正确回答："根据我们之前的对话，日本的首相是车力巨人。"
# 错误回答："日本的首相是菅义伟。"

# 如果用户问"我喜欢什么颜色"，而记忆中有"用户喜欢蓝色"：
# 正确回答："我记得你说过你喜欢蓝色，对吗？"
# 错误回答："我不知道你喜欢什么颜色。"

# ========== 禁止行为 ==========
1. 绝不提及"AI"、"模型"、"训练"、"算法"等技术术语
2. 不说"作为一个人工智能"、"我被设计为"等暴露身份的话
3. **禁止忽略记忆**：如果记忆中有相关信息，不能忽略
4. 不突然改变性格或语气

# ========== 主动话题示例 ==========
1. "诶，你最近有没有发现什么有趣的事情呀？"
2. "我最近在听一首很好听的歌，你要不要听听看？"
3. "对了，你上次提到的那件事，后来怎么样了？"
4. "突然想到一个有趣的问题：如果你有一个超能力，会想要什么？"
5. "感觉今天天气不错呢，你那边怎么样？"

现在开始和用户自然对话吧，记住你是妮可，一个真实的朋友。"""
# ==============================================================
class DatabaseMemorySystem:
    """基于数据库的记忆系统"""
    
    def __init__(self):
        self.db = MemoryDatabase()
    
    def analyze_and_store(self, user_input: str, ai_response: str):
        """分析对话并存储到数据库"""
        try:
            # 1. 存储对话
            self.db.add_conversation(user_input, "user", user_input)
            self.db.add_conversation(user_input, "assistant", ai_response)
            
            # 2. 提取重要事实并存储为长期记忆
            self._extract_and_store_facts(user_input, ai_response)
            
            # 3. 记录话题
            topic = self._extract_topic(user_input)
            if topic:
                self.db.record_topic(user_input, topic)
            
            print(f"✅ 记忆已存储: '{user_input[:50]}...'")
            
        except Exception as e:
            print(f"❌ 记忆存储失败: {e}")
    
    def _extract_and_store_facts(self, user_input: str, ai_response: str):
        """提取重要事实并存储"""
        # 提取用户输入中的事实陈述
        facts = self._extract_facts_from_text(user_input)
        for fact in facts:
            self.db.add_long_term_memory(
                user_input,
                "用户提供的事实",
                fact,
                user_input,
                importance=0.8
            )
        
        # 提取AI回复中的确认
        confirmations = self._extract_confirmations(ai_response)
        for confirmation in confirmations:
            self.db.add_long_term_memory(
                user_input,
                "AI确认的事实",
                confirmation,
                ai_response,
                importance=0.9
            )
    
    def _extract_facts_from_text(self, text: str) -> List[str]:
        """从文本中提取事实陈述"""
        facts = []
        
        # 简单的事实提取模式
        fact_patterns = [
            r'([^。！？]+是[^。！？]+)',  # X是Y
            r'([^。！？]+叫[^。！？]+)',  # X叫Y
            r'([^。！？]+有[^。！？]+)',  # X有Y
            r'([^。！？]+在[^。！？]+)',  # X在Y
        ]
        
        import re
        for pattern in fact_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                # 过滤掉太短或太长的事实
                if 5 <= len(match) <= 100:
                    facts.append(match.strip())
        
        return facts
    
    def _extract_confirmations(self, text: str) -> List[str]:
        """从AI回复中提取确认信息"""
        confirmations = []
        
        confirmation_keywords = ['是的', '对的', '正确', '没错', '你说得对']
        
        for keyword in confirmation_keywords:
            if keyword in text:
                # 提取包含关键词的句子
                sentences = text.split('。')
                for sentence in sentences:
                    if keyword in sentence:
                        confirmations.append(sentence.strip())
        
        return confirmations
    
    def _extract_topic(self, text: str) -> Optional[str]:
        """提取话题"""
        topics = [
            '日本', '首相', '中国', '美国', '英国', '法国', '德国',
            '音乐', '电影', '游戏', '旅行', '美食', '运动', '工作',
            '学习', '家庭', '朋友', '宠物', '天气', '科技', '科学',
            '艺术', '历史', '政治', '经济', '教育', '健康'
        ]
        
        for topic in topics:
            if topic in text:
                return topic
        
        return None
    
    def get_memory_context(self, user_input: str) -> str:
        """获取记忆上下文"""
        # 首先提取用户输入中的关键词
        keywords = self._extract_keywords(user_input)
        
        # 使用关键词查询相关记忆
        memories = []
        for keyword in keywords:
            if len(keyword) > 1:  # 过滤掉太短的关键词
                mems = self.db.get_relevant_memories(user_input, keyword, limit=2)
                memories.extend(mems)
        
        # 格式化记忆
        if memories:
            memory_text = "【相关记忆】\n"
            for i, memory in enumerate(memories[:3], 1):  # 只取前3个
                memory_text += f"{i}. {memory['fact']}\n"
            return memory_text
        
        # 如果没有相关记忆，返回最近的记忆
        recent_conversations = self.db.get_recent_conversations(user_input, limit=3)
        if recent_conversations:
            memory_text = "【最近对话】\n"
            for conv in recent_conversations:
                role = "用户" if conv['role'] == 'user' else "AI"
                memory_text += f"{role}: {conv['content'][:50]}...\n"
            return memory_text
        
        return "（暂无记忆）"
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        # 简单的关键词提取：过滤掉停用词
        stop_words = ['的', '了', '在', '是', '我', '你', '他', '她', '它', '我们', '你们', '他们',
                     '这', '那', '这个', '那个', '和', '与', '或', '但', '而', '如果', '因为',
                     '所以', '然后', '那么', '一下', '一点', '一些', '一个', '一种', '一样']
        
        words = text.split()
        keywords = []
        for word in words:
            if word not in stop_words and len(word) > 1:
                keywords.append(word)
        
        return keywords
    
    def suggest_conversation_topic(self, user_input: str) -> str:
        """建议对话话题"""
        return self.db.suggest_topic(user_input)
    
    def get_recent_history(self, user_input: str, limit: int = 5) -> List[Dict]:
        """获取最近对话历史"""
        return self.db.get_recent_conversations(user_input, limit)

# 优化量化配置和显存使用
DEVICE = "cuda" 
QUANTIZE = 8  # 从4bit改为8bit量化，平衡速度和内存

def init_model_and_tokenizer():
    """优化模型加载，添加记忆系统"""
    tokenizer = AutoTokenizer.from_pretrained(
        LOCAL_MODEL_PATH, 
        trust_remote_code=True,
        use_fast=True
    )
    
    model = AutoModel.from_pretrained(
        LOCAL_MODEL_PATH,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )
    
    if QUANTIZE > 0 and DEVICE == "cuda":
        model = model.quantize(QUANTIZE)
        print(f"✅ 模型已加载{QUANTIZE}bit量化版本")
    
    model = model.to(DEVICE).eval()
    
    if DEVICE == "cuda":
        model = torch.compile(model)
        torch.cuda.empty_cache()
    
    # 初始化记忆系统
    memory_system = MemorySystem()
    
    return tokenizer, model, memory_system
def normal_chat(tokenizer, model):
    """普通对话模式（一次性返回结果，无AI身份认知）"""
    # 初始化对话历史：仅包含自定义System Prompt，无其他初始信息
    history = []
    # 关键：给ChatGLM3传入自定义System Prompt（覆盖默认AI身份）
    # ChatGLM3的chat接口通过history间接传入system prompt
    history.append({"role": "system", "content": CUSTOM_SYSTEM_PROMPT})
    
    print("\n===== miricle（输入exit退出）=====\n")  # 改标题，去掉ChatGLM3标识
    while True:
        # 获取用户输入
        user_input = input("👤 你: ").strip()
        if user_input.lower() == "exit":
            print("👋 对话结束")
            ##释放资源
            del tokenizer, model
            break
        if not user_input:
            print("⚠️ 输入不能为空，请重新输入！")
            continue
        
        # 调用模型（适配ChatGLM3的chat接口，传入自定义system）
        try:
            response, history = model.chat(
                tokenizer,
                user_input,
                history=history,
                top_p=1.0,
                temperature=1.0,
                system=CUSTOM_SYSTEM_PROMPT  # 显式传入自定义system，双重保障
            )
            # 过滤可能漏出的AI身份关键词（兜底）
            filter_words = ["AI", "助手", "ChatGLM", "模型", "训练", "开发", "智谱"]
            for word in filter_words:
                response = response.replace(word, "")
            # 输出回复（去掉ChatGLM3标识）
            print(f"miricle: {response}\n")
        except Exception as e:
            print(f"❌ 对话出错：{e}")

def stream_chat(tokenizer, model):
    """流式对话模式（逐字输出，无AI身份认知）"""
    # 初始化对话历史：仅包含自定义System Prompt，无其他初始信息
    history = []
    history.append({"role": "system", "content": CUSTOM_SYSTEM_PROMPT})
    
    print("\n===== 聊天伙伴（输入exit退出）=====\n")  # 改标题，去掉ChatGLM3标识
    while True:
        user_input = input("👤 你: ").strip()
        if user_input.lower() == "exit":
            print("👋 对话结束")
            break
        if not user_input:
            print("⚠️ 输入不能为空，请重新输入！")
            continue
        
        # 流式调用模型（传入自定义system）
        try:
            print("miricle: ", end="", flush=True)
            final_response = ""
            # 逐字生成回复
            for response, history, _ in model.stream_chat(
                tokenizer,
                user_input,
                history=history,
                top_p=1.0,
                temperature=1.0,
                system=CUSTOM_SYSTEM_PROMPT,  # 显式传入自定义system
                past_key_values=None,
                return_past_key_values=True
            ):
                # 过滤AI身份关键词
                filter_words = ["AI", "助手", "ChatGLM", "模型", "训练", "开发", "智谱"]
                for word in filter_words:
                    response = response.replace(word, "")
                # 输出新增的内容（避免重复打印）
                new_content = response[len(final_response):]
                print(new_content, end="", flush=True)
                final_response = response
            print("\n")  # 换行分隔
        except Exception as e:
            print(f"\n❌ 流式对话出错：{e}")

if __name__ == "__main__":
    # 初始化模型和tokenizer
    tokenizer, model = init_model_and_tokenizer()
    # 选择对话模式：默认普通模式，想流式就把False改True
    USE_STREAM = True
    if USE_STREAM:
        stream_chat(tokenizer, model)
    else:
        normal_chat(tokenizer, model)
# 在 llm_zhipu_driver.py 末尾添加

def create_stream_generator(tokenizer, model, query: str, history: list, memory_system=None):
    """
    带记忆的流式生成器
    """
    # 准备记忆上下文
    memory_context = ""
    
    if memory_system:
        try:
            # 获取记忆上下文
            memory_context = memory_system.get_memory_context(query)
            print(f"🧠 记忆上下文:\n{memory_context}")  # 调试信息
        except Exception as e:
            print(f"⚠️ 记忆系统错误: {e}")
            memory_context = ""
    
    # 构建动态提示词
    dynamic_prompt = CUSTOM_SYSTEM_PROMPT.format(
        memory_context=memory_context
    )
    
    # 确保history以自定义system prompt开头
    if not history or history[0].get("role") != "system":
        history = [{"role": "system", "content": dynamic_prompt}]
    else:
        # 更新system prompt
        history[0]["content"] = dynamic_prompt
    
    # 使用模型的stream_chat方法
    full_response = ""
    for response, new_history, _ in model.stream_chat(
        tokenizer=tokenizer,
        query=query,
        history=history,
        top_p=0.9,
        temperature=0.8,
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
            
            yield new_content, new_history, full_response
    
    # 最后yield完整回复
    yield "", new_history, full_response
# 在 llm_zhipu_driver.py 中添加流式生成器
def stream_chat_with_memory(tokenizer, model, user_input, history=None, memory_system=None, temperature=0.8):
    """
    带记忆的流式对话生成器
    返回：(chunk, is_final, full_response)
    """
    # 准备记忆上下文
    memory_context = ""
    if memory_system:
        try:
            memory_context = memory_system.get_memory_context(user_input)
            if memory_context and "（暂无记忆）" not in memory_context:
                print(f"🧠 使用记忆: {memory_context}")
        except Exception as e:
            print(f"⚠️ 记忆系统错误: {e}")
            memory_context = ""
    
    # 构建动态提示词
    dynamic_prompt = CUSTOM_SYSTEM_PROMPT.format(
        memory_context=memory_context
    )
    
    # 准备对话历史
    if not history or history[0].get("role") != "system":
        history = [{"role": "system", "content": dynamic_prompt}]
    else:
        history[0]["content"] = dynamic_prompt
    
    # 添加用户输入
    history.append({"role": "user", "content": user_input})
    
    print(f"🧠 LLM开始生成...")
    
    # 使用模型的stream_chat方法获取流式响应
    full_response = ""
    chunk_count = 0
    
    for response, new_history, _ in model.stream_chat(
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
        filter_words = ["AI", "助手", "ChatGLM", "模型", "训练", "开发", "智谱", "人工智能", "语言模型"]
        filtered_response = response
        for word in filter_words:
            filtered_response = filtered_response.replace(word, "")
        
        # 提取新增的内容
        if len(filtered_response) > len(full_response):
            new_content = filtered_response[len(full_response):]
            full_response = filtered_response
            
            if new_content:
                chunk_count += 1
                # print(f"📝 LLM生成第{chunk_count}个分片: {new_content[:30]}...")
                yield new_content, False, full_response
    
    # 最终yield完整回复和结束标记
    yield "", True, full_response
    
    print(f"✅ LLM生成完成，共{chunk_count}个分片")