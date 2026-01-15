# memory_database.py
import sqlite3
import json
import time
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional
import hashlib

class MemoryDatabase:
    """记忆数据库管理类"""
    
    def __init__(self, db_path: str = "memory.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_db()
    
    def _init_db(self):
        """初始化数据库表"""
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # 1. 用户信息表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, key)
                )
            ''')
            
            # 2. 对话历史表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    tokens INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 3. 长期记忆表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS long_term_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    key_fact TEXT NOT NULL,
                    context TEXT,
                    importance REAL DEFAULT 0.5,
                    emotion_score REAL DEFAULT 0.0,
                    last_recalled TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    recall_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 4. 话题记录表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS topics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    subtopic TEXT,
                    interest_score REAL DEFAULT 0.5,
                    duration_seconds INTEGER DEFAULT 0,
                    talk_count INTEGER DEFAULT 1,
                    last_discussed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 5. 关系图谱表（实体关系）
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS entities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    entity_name TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    attributes TEXT,
                    relation_strength REAL DEFAULT 0.5,
                    last_mentioned TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    mention_count INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, entity_name)
                )
            ''')
            
            # 6. 情感记录表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS emotions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    emotion_type TEXT NOT NULL,
                    intensity REAL DEFAULT 0.5,
                    context TEXT,
                    duration_minutes INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 创建索引
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_conversations_user_session 
                ON conversations (user_id, session_id)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_conversations_created 
                ON conversations (created_at)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_long_term_memories_user_type 
                ON long_term_memories (user_id, memory_type)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_topics_user_topic 
                ON topics (user_id, topic)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_emotions_user_emotion 
                ON emotions (user_id, emotion_type)
            ''')
            
            conn.commit()
            conn.close()
    
    def _get_connection(self):
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # 启用字典形式返回
        return conn
    
    def _generate_user_id(self, user_input: str) -> str:
        """从用户输入生成用户ID（模拟）"""
        # 使用固定用户ID，这样所有对话都会关联到同一个用户
        return "default_user"
    
    # ========== 用户信息管理 ==========
    def update_user_profile(self, user_input: str, key: str, value: str, confidence: float = 1.0):
        """更新用户信息"""
        user_id = self._generate_user_id(user_input)
        
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO user_profiles 
                (user_id, key, value, confidence, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (user_id, key, value, confidence))
            
            conn.commit()
            conn.close()
        
        return True
    
    def get_user_profile(self, user_input: str) -> Dict[str, str]:
        """获取用户信息"""
        user_id = self._generate_user_id(user_input)
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT key, value, confidence FROM user_profiles 
            WHERE user_id = ? 
            ORDER BY confidence DESC
        ''', (user_id,))
        
        profile = {}
        for row in cursor.fetchall():
            profile[row['key']] = {
                'value': row['value'],
                'confidence': row['confidence']
            }
        
        conn.close()
        return profile
    
    # ========== 对话历史管理 ==========
    def add_conversation(self, user_input: str, role: str, content: str, 
                        session_id: str = "default"):
        """添加对话记录"""
        user_id = self._generate_user_id(user_input)
        
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # 计算token数（简化）
            tokens = len(content.split())
            
            cursor.execute('''
                INSERT INTO conversations 
                (user_id, role, content, session_id, tokens)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, role, content, session_id, tokens))
            
            conn.commit()
            conn.close()
        
        return True
    
    def get_recent_conversations(self, user_input: str, limit: int = 10) -> List[Dict]:
        """获取最近的对话记录"""
        user_id = self._generate_user_id(user_input)
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT role, content, created_at 
            FROM conversations 
            WHERE user_id = ? 
            ORDER BY created_at DESC 
            LIMIT ?
        ''', (user_id, limit))
        
        conversations = []
        for row in cursor.fetchall():
            conversations.append({
                'role': row['role'],
                'content': row['content'],
                'time': row['created_at']
            })
        
        conn.close()
        return conversations
    
    # ========== 长期记忆管理 ==========
    def add_long_term_memory(self, user_input: str, memory_type: str, 
                           key_fact: str, context: str = None, 
                           importance: float = 0.5):
        """添加长期记忆"""
        user_id = self._generate_user_id(user_input)
        
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # 检查是否已存在相似记忆
            cursor.execute('''
                SELECT id FROM long_term_memories 
                WHERE user_id = ? AND key_fact = ? AND memory_type = ?
            ''', (user_id, key_fact, memory_type))
            
            existing = cursor.fetchone()
            
            if existing:
                # 更新现有记忆
                cursor.execute('''
                    UPDATE long_term_memories 
                    SET importance = ?, updated_at = CURRENT_TIMESTAMP,
                        recall_count = recall_count + 1
                    WHERE id = ?
                ''', (importance, existing['id']))
            else:
                # 插入新记忆
                cursor.execute('''
                    INSERT INTO long_term_memories 
                    (user_id, memory_type, key_fact, context, importance)
                    VALUES (?, ?, ?, ?, ?)
                ''', (user_id, memory_type, key_fact, context, importance))
            
            conn.commit()
            conn.close()
        
        return True
    
    def get_relevant_memories(self, user_input: str, query: str = None, 
                            limit: int = 5) -> List[Dict]:
        """获取相关记忆（基于关键词匹配）"""
        user_id = self._generate_user_id(user_input)
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if query:
            # 简单的关键词匹配
            keywords = query.split()
            conditions = []
            params = [user_id]
            
            for keyword in keywords:
                conditions.append("(key_fact LIKE ? OR context LIKE ?)")
                params.extend([f"%{keyword}%", f"%{keyword}%"])
            
            where_clause = " AND ".join(conditions)
            sql = f'''
                SELECT memory_type, key_fact, context, importance, 
                       recall_count, last_recalled
                FROM long_term_memories 
                WHERE user_id = ? AND ({where_clause})
                ORDER BY importance DESC, recall_count DESC, last_recalled DESC
                LIMIT ?
            '''
            params.append(limit)
            
            cursor.execute(sql, params)
        else:
            # 获取最重要的记忆
            cursor.execute('''
                SELECT memory_type, key_fact, context, importance, 
                       recall_count, last_recalled
                FROM long_term_memories 
                WHERE user_id = ?
                ORDER BY importance DESC, recall_count DESC, last_recalled DESC
                LIMIT ?
            ''', (user_id, limit))
        
        memories = []
        for row in cursor.fetchall():
            memories.append({
                'type': row['memory_type'],
                'fact': row['key_fact'],
                'context': row['context'],
                'importance': row['importance'],
                'recall_count': row['recall_count'],
                'last_recalled': row['last_recalled']
            })
        
        conn.close()
        return memories
    
    # ========== 话题管理 ==========
    def record_topic(self, user_input: str, topic: str, subtopic: str = None,
                    interest_score: float = 0.5, duration: int = 0):
        """记录讨论的话题"""
        user_id = self._generate_user_id(user_input)
        
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO topics 
                (user_id, topic, subtopic, interest_score, duration_seconds, 
                 talk_count, last_discussed)
                VALUES (?, ?, ?, ?, ?, 
                        COALESCE((SELECT talk_count FROM topics 
                                  WHERE user_id = ? AND topic = ?), 0) + 1,
                        CURRENT_TIMESTAMP)
            ''', (user_id, topic, subtopic, interest_score, duration,
                  user_id, topic))
            
            conn.commit()
            conn.close()
        
        return True
    
    def get_favorite_topics(self, user_input: str, limit: int = 5) -> List[Dict]:
        """获取用户喜欢的话题"""
        user_id = self._generate_user_id(user_input)
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT topic, subtopic, interest_score, talk_count, last_discussed
            FROM topics 
            WHERE user_id = ?
            ORDER BY interest_score DESC, talk_count DESC
            LIMIT ?
        ''', (user_id, limit))
        
        topics = []
        for row in cursor.fetchall():
            topics.append({
                'topic': row['topic'],
                'subtopic': row['subtopic'],
                'interest': row['interest_score'],
                'talk_count': row['talk_count'],
                'last_discussed': row['last_discussed']
            })
        
        conn.close()
        return topics
    
    def suggest_topic(self, user_input: str) -> str:
        """推荐话题"""
        favorite_topics = self.get_favorite_topics(user_input, 3)
        
        if not favorite_topics:
            # 默认话题
            default_topics = [
                "音乐",
                "电影",
                "旅行",
                "美食",
                "运动",
                "阅读",
                "游戏"
            ]
            import random
            return random.choice(default_topics)
        
        # 选择最久没讨论的话题
        import random
        weights = []
        for topic in favorite_topics:
            # 计算时间权重（越久没讨论权重越高）
            time_weight = random.random() * 0.5  # 随机成分
            weights.append(time_weight)
        
        # 加权随机选择
        total_weight = sum(weights)
        rand_val = random.random() * total_weight
        
        cumulative = 0
        for i, weight in enumerate(weights):
            cumulative += weight
            if rand_val <= cumulative:
                return favorite_topics[i]['topic']
        
        return favorite_topics[0]['topic']
    
    # ========== 工具函数 ==========
    def get_memory_summary(self, user_input: str) -> Dict:
        """获取记忆摘要（用于提示词）"""
        user_id = self._generate_user_id(user_input)
        
        summary = {
            'user_profile': self.get_user_profile(user_input),
            'recent_memories': self.get_relevant_memories(user_input, limit=3),
            'favorite_topics': self.get_favorite_topics(user_input, limit=3)
        }
        
        return summary
    
    def format_memory_for_prompt(self, user_input: str) -> str:
        """将记忆格式化为提示词"""
        summary = self.get_memory_summary(user_input)
        
        prompt_parts = []
        
        # 1. 用户信息
        if summary['user_profile']:
            prompt_parts.append("【用户信息】")
            for key, info in summary['user_profile'].items():
                prompt_parts.append(f"- {key}: {info['value']}")
        
        # 2. 重要记忆
        if summary['recent_memories']:
            prompt_parts.append("\n【重要记忆】")
            for memory in summary['recent_memories']:
                fact = memory['fact']
                context = f"({memory['context']})" if memory['context'] else ""
                prompt_parts.append(f"- {fact} {context}")
        
        # 3. 喜好话题
        if summary['favorite_topics']:
            prompt_parts.append("\n【喜好话题】")
            for topic in summary['favorite_topics']:
                count = topic['talk_count']
                prompt_parts.append(f"- {topic['topic']}（讨论过{count}次）")
        
        return "\n".join(prompt_parts) if prompt_parts else "（暂无记忆）"
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """清理旧数据"""
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # 清理旧的对话记录（保留30天）
            cursor.execute('''
                DELETE FROM conversations 
                WHERE DATE(created_at) < DATE('now', ?)
            ''', (f'-{days_to_keep} days',))
            
            # 清理旧的情感记录
            cursor.execute('''
                DELETE FROM emotions 
                WHERE DATE(created_at) < DATE('now', ?)
            ''', (f'-{days_to_keep} days',))
            
            conn.commit()
            conn.close()
        
        return True