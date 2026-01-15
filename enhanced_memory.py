"""
增强记忆模块
为LLM对话提供持久化、精准的记忆功能
"""

import re
import time
import json
import sqlite3
import hashlib
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from datetime import datetime, timedelta

class FactExtractor:
    """事实提取器：从文本中提取结构化事实"""
    
    def __init__(self):
        self.patterns = {
            'is_a': [  # X是Y
                r'([^，。！？]+是[^，。！？]+)',
                r'([^，。！？]+就是[^，。！？]+)',
                r'([^，。！？]+为[^，。！？]+)',
            ],
            'has': [  # X有Y
                r'([^，。！？]+有[^，。！？]+)',
                r'([^，。！？]+拥有[^，。！？]+)',
                r'([^，。！？]+具备[^，。！？]+)',
            ],
            'like': [  # X喜欢Y
                r'([^，。！？]+喜欢[^，。！？]+)',
                r'([^，。！？]+爱[^，。！？]+)',
                r'([^，。！？]+热爱[^，。！？]+)',
            ],
            'at': [  # X在Y
                r'([^，。！？]+在[^，。！？]+)',
                r'([^，。！？]+位于[^，。！？]+)',
            ],
            'do': [  # X做Y
                r'([^，。！？]+做[^，。！？]+)',
                r'([^，。！？]+从事[^，。！？]+)',
            ]
        }
        
        self.stop_words = {'我', '你', '他', '她', '它', '我们', '你们', '他们', 
                          '这个', '那个', '这些', '那些', '现在', '今天', '昨天',
                          '明天', '刚才', '然后', '所以', '因为', '但是', '而且'}
    
    def extract_facts(self, text: str) -> List[Dict]:
        """从文本中提取事实"""
        facts = []
        
        for fact_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    # 清理事实文本
                    clean_fact = self._clean_fact_text(match)
                    if clean_fact and len(clean_fact) >= 4:
                        facts.append({
                            'type': fact_type,
                            'fact': clean_fact,
                            'confidence': 0.8,
                            'raw_text': match
                        })
        
        return facts
    
    def _clean_fact_text(self, text: str) -> str:
        """清理事实文本"""
        # 移除句首的停止词
        words = text.split()
        while words and words[0] in self.stop_words:
            words.pop(0)
        
        # 移除句尾的标点
        while words and words[-1] in {'。', '！', '？', '.', '!', '?', '，', ','}:
            words.pop(-1)
        
        return ' '.join(words)
    
    def extract_entities(self, text: str) -> List[str]:
        """提取实体"""
        entities = []
        
        # 简单实体提取：名词性词组
        noun_patterns = [
            r'(\w+总理)', r'(\w+总统)', r'(\w+主席)', r'(\w+国王)',  # 职位
            r'(\w+人)', r'(\w+国)', r'(\w+市)', r'(\w+省)',  # 地域
            r'(\w+公司)', r'(\w+大学)', r'(\w+学校)',  # 机构
        ]
        
        for pattern in noun_patterns:
            matches = re.findall(pattern, text)
            entities.extend(matches)
        
        return entities

class MemoryDatabase:
    """记忆数据库"""
    
    def __init__(self, db_path: str = "enhanced_memory.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 事实表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact_hash TEXT UNIQUE NOT NULL,
                fact_text TEXT NOT NULL,
                fact_type TEXT NOT NULL,
                entity TEXT,
                predicate TEXT,
                confidence REAL DEFAULT 0.8,
                source_text TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_recalled TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                recall_count INTEGER DEFAULT 0,
                is_active BOOLEAN DEFAULT 1
            )
        ''')
        
        # 实体表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_name TEXT UNIQUE NOT NULL,
                entity_type TEXT,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 对话上下文表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS context (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                query TEXT NOT NULL,
                response TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 创建索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_facts_entity ON facts(entity)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_facts_type ON facts(fact_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_context_session ON context(session_id)')
        
        conn.commit()
        conn.close()
    
    def store_fact(self, fact_text: str, fact_type: str, entity: str = None, 
                   predicate: str = None, confidence: float = 0.8, source_text: str = None):
        """存储事实"""
        fact_hash = hashlib.md5(fact_text.encode()).hexdigest()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO facts 
                (fact_hash, fact_text, fact_type, entity, predicate, confidence, source_text, last_recalled)
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (fact_hash, fact_text, fact_type, entity, predicate, confidence, source_text))
            
            conn.commit()
            return True
        except Exception as e:
            print(f"存储事实失败: {e}")
            return False
        finally:
            conn.close()
    
    def get_relevant_facts(self, query: str, limit: int = 5) -> List[Dict]:
        """获取相关事实"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # 提取查询中的关键词
        keywords = query.split()
        conditions = []
        params = []
        
        for keyword in keywords:
            if len(keyword) > 1:  # 过滤掉单字
                conditions.append("(fact_text LIKE ? OR entity LIKE ?)")
                params.extend([f"%{keyword}%", f"%{keyword}%"])
        
        if conditions:
            sql = f'''
                SELECT fact_text, fact_type, entity, predicate, confidence, last_recalled, recall_count
                FROM facts 
                WHERE is_active = 1 AND ({' OR '.join(conditions)})
                ORDER BY confidence DESC, recall_count DESC, last_recalled DESC
                LIMIT ?
            '''
            params.append(limit)
            cursor.execute(sql, params)
        else:
            # 如果没有关键词，返回最近使用的事实
            cursor.execute('''
                SELECT fact_text, fact_type, entity, predicate, confidence, last_recalled, recall_count
                FROM facts 
                WHERE is_active = 1
                ORDER BY last_recalled DESC, confidence DESC
                LIMIT ?
            ''', (limit,))
        
        facts = []
        for row in cursor.fetchall():
            facts.append({
                'text': row['fact_text'],
                'type': row['fact_type'],
                'entity': row['entity'],
                'predicate': row['predicate'],
                'confidence': row['confidence'],
                'last_recalled': row['last_recalled'],
                'recall_count': row['recall_count']
            })
        
        conn.close()
        return facts
    
    def mark_fact_recalled(self, fact_text: str):
        """标记事实被回忆"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE facts 
            SET recall_count = recall_count + 1, last_recalled = CURRENT_TIMESTAMP
            WHERE fact_text = ?
        ''', (fact_text,))
        
        conn.commit()
        conn.close()
    
    def store_conversation(self, session_id: str, query: str, response: str):
        """存储对话"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO context (session_id, query, response)
            VALUES (?, ?, ?)
        ''', (session_id, query, response))
        
        conn.commit()
        conn.close()
    
    def get_recent_conversations(self, session_id: str, limit: int = 3) -> List[Dict]:
        """获取最近的对话"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT query, response, timestamp
            FROM context 
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (session_id, limit))
        
        conversations = []
        for row in cursor.fetchall():
            conversations.append({
                'query': row['query'],
                'response': row['response'],
                'time': row['timestamp']
            })
        
        conn.close()
        return conversations

class EnhancedMemorySystem:
    """增强记忆系统"""
    
    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.fact_extractor = FactExtractor()
        self.database = MemoryDatabase()
        self.session_id = f"session_{int(time.time())}"
        
        # 短期记忆缓存
        self.short_term_memory = []
        self.short_term_limit = 10
        
        # 实体追踪
        self.entity_facts = defaultdict(list)
    
    def process_conversation(self, user_input: str, ai_response: str):
        """处理对话，提取和存储记忆"""
        # 1. 存储对话上下文
        self.database.store_conversation(self.session_id, user_input, ai_response)
        
        # 2. 从用户输入中提取事实
        user_facts = self.fact_extractor.extract_facts(user_input)
        for fact in user_facts:
            # 尝试提取实体和谓词
            entity, predicate = self._extract_entity_predicate(fact['fact'])
            
            self.database.store_fact(
                fact_text=fact['fact'],
                fact_type=fact['type'],
                entity=entity,
                predicate=predicate,
                confidence=fact['confidence'],
                source_text=user_input
            )
            
            if entity:
                self.entity_facts[entity].append(fact['fact'])
        
        # 3. 从AI回复中提取确认
        if any(word in ai_response for word in ['是的', '对的', '正确', '没错', '你说得对']):
            ai_facts = self.fact_extractor.extract_facts(ai_response)
            for fact in ai_facts:
                entity, predicate = self._extract_entity_predicate(fact['fact'])
                
                self.database.store_fact(
                    fact_text=fact['fact'],
                    fact_type=f"{fact['type']}_confirmed",
                    entity=entity,
                    predicate=predicate,
                    confidence=0.9,  # AI确认的事实置信度更高
                    source_text=ai_response
                )
        
        # 4. 更新短期记忆
        self.short_term_memory.append({
            'user': user_input,
            'ai': ai_response,
            'time': time.time()
        })
        if len(self.short_term_memory) > self.short_term_limit:
            self.short_term_memory = self.short_term_memory[-self.short_term_limit:]
    
    def _extract_entity_predicate(self, fact_text: str) -> Tuple[Optional[str], Optional[str]]:
        """从事实中提取实体和谓词"""
        if '是' in fact_text:
            parts = fact_text.split('是', 1)
            if len(parts) == 2:
                return parts[0].strip(), parts[1].strip()
        elif '有' in fact_text:
            parts = fact_text.split('有', 1)
            if len(parts) == 2:
                return parts[0].strip(), parts[1].strip()
        elif '在' in fact_text:
            parts = fact_text.split('在', 1)
            if len(parts) == 2:
                return parts[0].strip(), parts[1].strip()
        
        return None, None
    
    def get_memory_context(self, query: str) -> str:
        """获取记忆上下文"""
        context_parts = []
        
        # 1. 获取相关事实
        relevant_facts = self.database.get_relevant_facts(query, limit=3)
        if relevant_facts:
            context_parts.append("【重要事实】")
            for i, fact in enumerate(relevant_facts, 1):
                # 标记事实被回忆
                self.database.mark_fact_recalled(fact['text'])
                
                # 格式化事实显示
                fact_display = fact['text']
                if fact['confidence'] < 0.7:
                    fact_display += "（不确定）"
                
                context_parts.append(f"{i}. {fact_display}")
        
        # 2. 获取最近对话
        recent_convs = self.database.get_recent_conversations(self.session_id, limit=2)
        if recent_convs and len(context_parts) < 3:  # 如果事实太少，添加对话
            context_parts.append("\n【最近对话】")
            for conv in recent_convs:
                context_parts.append(f"用户: {conv['query'][:50]}...")
                context_parts.append(f"我: {conv['response'][:50]}...")
        
        # 3. 短期记忆
        if self.short_term_memory and len(context_parts) < 4:
            context_parts.append("\n【短期记忆】")
            for mem in self.short_term_memory[-2:]:
                context_parts.append(f"- {mem['user'][:30]}...")
        
        if context_parts:
            return "\n".join(context_parts)
        return "（暂无记忆）"
    
    def get_facts_by_entity(self, entity: str) -> List[str]:
        """获取实体的所有事实"""
        if entity in self.entity_facts:
            return self.entity_facts[entity]
        
        # 从数据库查询
        facts = self.database.get_relevant_facts(entity, limit=10)
        return [fact['text'] for fact in facts]
    
    def clear_short_term_memory(self):
        """清空短期记忆"""
        self.short_term_memory = []
    
    def export_memory(self, filepath: str = "memory_export.json"):
        """导出记忆到文件"""
        # 获取所有事实
        conn = sqlite3.connect(self.database.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT fact_text, fact_type, entity, predicate, confidence, created_at, recall_count
            FROM facts 
            WHERE is_active = 1
            ORDER BY confidence DESC, recall_count DESC
        ''')
        
        facts_data = []
        for row in cursor.fetchall():
            facts_data.append(dict(row))
        
        conn.close()
        
        # 导出数据
        export_data = {
            'user_id': self.user_id,
            'export_time': datetime.now().isoformat(),
            'facts_count': len(facts_data),
            'facts': facts_data,
            'entity_summary': {entity: len(facts) for entity, facts in self.entity_facts.items()}
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        return filepath

class EnhancedMemoryLLM:
    """增强记忆的LLM包装器"""
    
    def __init__(self, base_model, tokenizer, user_id: str = "default"):
        self.model = base_model
        self.tokenizer = tokenizer
        self.memory_system = EnhancedMemorySystem(user_id)
        self.conversation_history = []
        
        # 生成参数
        self.generation_config = {
            'temperature': 0.2,  # 低温度，更确定性
            'top_p': 0.7,
            'repetition_penalty': 1.1,
            'max_length': 512
        }
    
    def create_memory_prompt(self, query: str) -> str:
        """创建带记忆的提示词"""
        memory_context = self.memory_system.get_memory_context(query)
        
        # 构建强化提示词
        prompt = f"""你叫妮可(Nicole)，一个活泼开朗、善于倾听的虚拟朋友。

# 重要指令
你必须根据以下记忆来回答问题。这些记忆来自你和用户的对话历史。
**如果记忆中有相关信息，你必须优先使用记忆中的信息，而不是你已有的知识。**

# 记忆内容
{memory_context}

# 用户问题
{query}

# 回答要求
1. 如果记忆中有答案，直接使用记忆中的信息
2. 可以引用记忆，例如"我记得你告诉过我..."
3. 保持自然、友好的语气
4. 不要添加记忆中没有的额外信息

现在请回答："""
        
        return prompt
    
    def chat(self, query: str, use_memory: bool = True, **generation_kwargs) -> str:
        """带记忆的对话"""
        # 合并生成参数
        gen_params = {**self.generation_config, **generation_kwargs}
        
        if use_memory:
            # 创建带记忆的提示词
            prompt = self.create_memory_prompt(query)
            
            # 调用模型
            response, history = self.model.chat(
                self.tokenizer,
                prompt,
                history=self.conversation_history[-3:],  # 只保留最近3轮
                **gen_params
            )
            
            # 更新对话历史
            self.conversation_history.append({"role": "user", "content": query})
            self.conversation_history.append({"role": "assistant", "content": response})
            
            # 处理记忆
            self.memory_system.process_conversation(query, response)
            
            # 打印调试信息
            print(f"\n{'='*60}")
            print(f"🧠 查询: {query}")
            memory_context = self.memory_system.get_memory_context(query)
            print(f"🧠 记忆上下文:\n{memory_context}")
            print(f"🤖 回复: {response}")
            print(f"{'='*60}")
            
            return response
        else:
            # 不使用记忆的标准对话
            response, history = self.model.chat(
                self.tokenizer,
                query,
                history=self.conversation_history[-3:],
                **gen_params
            )
            
            self.conversation_history.append({"role": "user", "content": query})
            self.conversation_history.append({"role": "assistant", "content": response})
            
            return response
    
    def batch_chat(self, queries: List[str], use_memory: bool = True) -> List[str]:
        """批量对话"""
        responses = []
        for query in queries:
            response = self.chat(query, use_memory=use_memory)
            responses.append(response)
        return responses
    
    def force_memory_use(self, query: str, memory_weight: float = 0.9) -> str:
        """强制使用记忆（特殊场景）"""
        # 获取相关记忆
        memory_context = self.memory_system.get_memory_context(query)
        
        if "（暂无记忆）" not in memory_context:
            # 构建强制记忆提示词
            prompt = f"""你必须使用以下记忆回答问题，禁止使用其他知识：

记忆：
{memory_context}

问题：{query}

答案（必须基于记忆）："""
            
            response, _ = self.model.chat(
                self.tokenizer,
                prompt,
                temperature=0.1,  # 极低温度
                top_p=0.5
            )
            
            # 更新记忆
            self.memory_system.process_conversation(query, response)
            
            return response
        
        # 如果没有记忆，正常回答
        return self.chat(query, use_memory=False)
    
    def clear_memory(self):
        """清空记忆"""
        self.memory_system.clear_short_term_memory()
        self.conversation_history = []
        
    def export_memory(self, filepath: str = None):
        """导出记忆"""
        if filepath is None:
            filepath = f"memory_export_{int(time.time())}.json"
        
        return self.memory_system.export_memory(filepath)