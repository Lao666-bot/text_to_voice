# memory_query.py
import sqlite3
import json
from typing import List, Dict

class MemoryQuery:
    """记忆查询接口"""
    
    def __init__(self, db_path="memory.db"):
        self.db_path = db_path
    
    def search_conversations(self, keyword: str, limit: int = 10) -> List[Dict]:
        """搜索对话记录"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        cursor = conn.cursor()
        cursor.execute('''
            SELECT role, content, created_at 
            FROM conversations 
            WHERE content LIKE ? 
            ORDER BY created_at DESC 
            LIMIT ?
        ''', (f"%{keyword}%", limit))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'role': row['role'],
                'content': row['content'],
                'time': row['created_at']
            })
        
        conn.close()
        return results
    
    def get_user_timeline(self, user_id: str = None) -> List[Dict]:
        """获取用户时间线"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        cursor = conn.cursor()
        
        if user_id:
            cursor.execute('''
                SELECT 
                    'conversation' as type,
                    content as description,
                    created_at
                FROM conversations 
                WHERE user_id = ?
                
                UNION ALL
                
                SELECT 
                    'memory' as type,
                    key_fact as description,
                    created_at
                FROM long_term_memories 
                WHERE user_id = ?
                
                UNION ALL
                
                SELECT 
                    'emotion' as type,
                    emotion_type as description,
                    created_at
                FROM emotions 
                WHERE user_id = ?
                
                ORDER BY created_at DESC
                LIMIT 50
            ''', (user_id, user_id, user_id))
        else:
            cursor.execute('''
                SELECT 
                    'conversation' as type,
                    content as description,
                    created_at,
                    user_id
                FROM conversations 
                
                UNION ALL
                
                SELECT 
                    'memory' as type,
                    key_fact as description,
                    created_at,
                    user_id
                FROM long_term_memories 
                
                UNION ALL
                
                SELECT 
                    'emotion' as type,
                    emotion_type as description,
                    created_at,
                    user_id
                FROM emotions 
                
                ORDER BY created_at DESC
                LIMIT 50
            ''')
        
        timeline = []
        for row in cursor.fetchall():
            item = {
                'type': row['type'],
                'description': row['description'],
                'time': row['created_at']
            }
            if 'user_id' in row.keys():
                item['user_id'] = row['user_id']
            timeline.append(item)
        
        conn.close()
        return timeline
    
    def get_relationship_graph(self, user_id: str) -> Dict:
        """获取关系图谱"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        cursor = conn.cursor()
        cursor.execute('''
            SELECT entity_name, entity_type, relation_strength, mention_count
            FROM entities 
            WHERE user_id = ?
            ORDER BY relation_strength DESC
        ''', (user_id,))
        
        nodes = []
        links = []
        
        for row in cursor.fetchall():
            nodes.append({
                'id': row['entity_name'],
                'type': row['entity_type'],
                'strength': row['relation_strength'],
                'mentions': row['mention_count']
            })
            
            # 简单的链接生成（实际应该根据上下文分析关系）
            if len(nodes) > 1:
                links.append({
                    'source': nodes[0]['id'],
                    'target': row['entity_name'],
                    'value': row['relation_strength']
                })
        
        conn.close()
        
        return {
            'nodes': nodes,
            'links': links
        }
    
    def export_for_training(self, output_file="training_data.json"):
        """导出用于模型训练的数据"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        cursor = conn.cursor()
        cursor.execute('''
            SELECT c1.content as user_input, c2.content as assistant_response
            FROM conversations c1
            JOIN conversations c2 ON c1.user_id = c2.user_id 
                AND c1.session_id = c2.session_id 
                AND c1.created_at < c2.created_at
            WHERE c1.role = 'user' AND c2.role = 'assistant'
            ORDER BY c1.created_at
            LIMIT 1000
        ''')
        
        training_data = []
        for row in cursor.fetchall():
            training_data.append({
                'user': row['user_input'],
                'assistant': row['assistant_response']
            })
        
        conn.close()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        
        return len(training_data)