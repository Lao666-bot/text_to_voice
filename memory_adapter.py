"""
记忆适配器：将增强记忆模块集成到现有系统
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_memory import EnhancedMemoryLLM
from typing import Optional

class MemoryAdapter:
    """记忆适配器：桥接现有系统和增强记忆模块"""
    
    def __init__(self, base_model, tokenizer):
        """初始化适配器"""
        print("🚀 初始化增强记忆适配器...")
        
        # 创建增强记忆LLM
        self.enhanced_llm = EnhancedMemoryLLM(base_model, tokenizer)
        
        # 状态跟踪
        self.conversation_count = 0
        self.memory_hits = 0
        self.last_query = ""
        
        print("✅ 增强记忆适配器初始化完成")
    
    def process_query(self, query: str, use_memory: bool = True, 
                     force_memory: bool = False, **kwargs) -> str:
        """处理查询"""
        self.conversation_count += 1
        self.last_query = query
        
        # 检查是否应该强制使用记忆
        should_force = force_memory or self._should_force_memory(query)
        
        if should_force:
            response = self.enhanced_llm.force_memory_use(query)
            self.memory_hits += 1
        elif use_memory:
            response = self.enhanced_llm.chat(query, **kwargs)
        else:
            response = self.enhanced_llm.chat(query, use_memory=False, **kwargs)
        
        # 检查是否使用了记忆
        memory_context = self.enhanced_llm.memory_system.get_memory_context(query)
        if "（暂无记忆）" not in memory_context:
            self.memory_hits += 1
        
        return response
    
    def _should_force_memory(self, query: str) -> bool:
        """判断是否应该强制使用记忆"""
        # 检查是否是重复问题
        if query == self.last_query:
            return True
        
        # 检查是否包含特定关键词
        force_keywords = ['之前说过', '刚才说', '记得吗', '还记得吗', '说过']
        for keyword in force_keywords:
            if keyword in query:
                return True
        
        return False
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            'conversation_count': self.conversation_count,
            'memory_hits': self.memory_hits,
            'memory_hit_rate': self.memory_hits / max(self.conversation_count, 1),
            'short_term_memory_size': len(self.enhanced_llm.memory_system.short_term_memory)
        }
    
    def clear_memory(self):
        """清空记忆"""
        self.enhanced_llm.clear_memory()
        self.conversation_count = 0
        self.memory_hits = 0
        self.last_query = ""
        print("🧹 记忆已清空")
    
    def export_memory(self, filepath: str = None):
        """导出记忆"""
        return self.enhanced_llm.export_memory(filepath)
    
    def get_memory_context(self, query: str) -> str:
        """获取记忆上下文"""
        return self.enhanced_llm.memory_system.get_memory_context(query)
    
    def get_facts_by_entity(self, entity: str) -> list:
        """获取实体的所有事实"""
        return self.enhanced_llm.memory_system.get_facts_by_entity(entity)
    
    def manual_add_fact(self, fact_text: str, fact_type: str = "manual"):
        """手动添加事实"""
        self.enhanced_llm.memory_system.process_conversation(
            f"手动添加事实: {fact_text}",
            f"已确认事实: {fact_text}"
        )
        print(f"📝 手动添加事实: {fact_text}")
def process_query_stream(self, user_input: str, use_memory: bool = True, 
                         force_memory: bool = False, temperature: float = 0.8):
    """流式处理查询，返回生成器"""
    # 准备记忆上下文
    memory_context = ""
    memory_hit = False
    
    if use_memory:
        try:
            memory_context = self.memory_system.get_memory_context(user_input)
            if memory_context and "（暂无记忆）" not in memory_context:
                memory_hit = True
        except Exception as e:
            print(f"⚠️ 记忆系统错误: {e}")
            memory_context = ""
    
    # 构建动态提示词
    dynamic_prompt = self.system_prompt.format(
        memory_context=memory_context
    )
    
    # 准备对话历史
    if not self.history or self.history[0].get("role") != "system":
        self.history = [{"role": "system", "content": dynamic_prompt}]
    else:
        self.history[0]["content"] = dynamic_prompt
    
    # 使用模型的stream_chat方法
    full_response = ""
    for response, new_history, _ in self.model.stream_chat(
        tokenizer=self.tokenizer,
        query=user_input,
        history=self.history,
        top_p=0.9,
        temperature=temperature,
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
            
            yield new_content, False
    
    # 最终标记
    yield "", True
    
    # 更新历史
    self.history = [{"role": "system", "content": dynamic_prompt},
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": full_response}]
    
    # 更新统计
    self.conversation_count += 1
    if memory_hit:
        self.memory_hits += 1
    
    # 存储到记忆系统
    if use_memory and self.memory_system:
        self.memory_system.analyze_and_store(user_input, full_response)