# memory_analyzer.py
import sqlite3
import json
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

class MemoryAnalyzer:
    """记忆数据分析器"""
    
    def __init__(self, db_path="memory.db"):
        self.db_path = db_path
    
    def get_statistics(self):
        """获取数据库统计信息"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        stats = {}
        
        # 表信息
        tables = ['conversations', 'user_profiles', 'long_term_memories', 
                  'topics', 'entities', 'emotions']
        
        for table in tables:
            cursor = conn.cursor()
            cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
            row = cursor.fetchone()
            stats[table] = row['count'] if row else 0
        
        # 对话统计
        cursor = conn.cursor()
        cursor.execute('''
            SELECT 
                COUNT(*) as total_messages,
                COUNT(DISTINCT user_id) as unique_users,
                AVG(tokens) as avg_tokens,
                MAX(created_at) as last_message
            FROM conversations
        ''')
        row = cursor.fetchone()
        stats['conversations'] = {
            'total_messages': row['total_messages'],
            'unique_users': row['unique_users'],
            'avg_tokens': row['avg_tokens'],
            'last_message': row['last_message']
        }
        
        conn.close()
        return stats
    
    def export_conversations(self, output_file="conversations.json"):
        """导出对话历史"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        cursor = conn.cursor()
        cursor.execute('''
            SELECT user_id, role, content, created_at, session_id
            FROM conversations
            ORDER BY created_at
        ''')
        
        conversations = []
        for row in cursor.fetchall():
            conversations.append({
                'user_id': row['user_id'],
                'role': row['role'],
                'content': row['content'],
                'timestamp': row['created_at'],
                'session_id': row['session_id']
            })
        
        conn.close()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(conversations, f, ensure_ascii=False, indent=2)
        
        return len(conversations)
    
    def analyze_topics(self):
        """分析话题趋势"""
        conn = sqlite3.connect(self.db_path)
        
        # 获取话题数据
        query = '''
            SELECT 
                topic,
                COUNT(*) as discussion_count,
                AVG(interest_score) as avg_interest,
                MAX(last_discussed) as last_discussed
            FROM topics
            GROUP BY topic
            ORDER BY discussion_count DESC
        '''
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df
    
    def plot_memory_growth(self):
        """绘制记忆增长图"""
        conn = sqlite3.connect(self.db_path)
        
        # 获取按日统计的记忆增长
        query = '''
            SELECT 
                DATE(created_at) as date,
                COUNT(*) as memory_count
            FROM long_term_memories
            GROUP BY DATE(created_at)
            ORDER BY date
        '''
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if not df.empty:
            plt.figure(figsize=(10, 6))
            plt.plot(df['date'], df['memory_count'], marker='o')
            plt.xlabel('日期')
            plt.ylabel('记忆数量')
            plt.title('长期记忆增长趋势')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('memory_growth.png')
            plt.close()
    
    def cleanup_database(self, days_to_keep=30):
        """清理数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        tables = ['conversations', 'emotions']
        
        for table in tables:
            cursor.execute(f'''
                DELETE FROM {table} 
                WHERE DATE(created_at) < DATE('now', ?)
            ''', (f'-{days_to_keep} days',))
        
        conn.commit()
        
        # 清理后优化数据库
        cursor.execute("VACUUM")
        conn.close()
        
        return True
    
    def backup_database(self, backup_path="memory_backup.db"):
        """备份数据库"""
        import shutil
        shutil.copy2(self.db_path, backup_path)
        return backup_path

# 使用示例
if __name__ == "__main__":
    analyzer = MemoryAnalyzer()
    
    print("=== 记忆数据库分析 ===")
    stats = analyzer.get_statistics()
    
    for table, count in stats.items():
        if isinstance(count, dict):
            print(f"\n{table}:")
            for k, v in count.items():
                print(f"  {k}: {v}")
        else:
            print(f"{table}: {count}条记录")
    
    # 导出对话
    export_count = analyzer.export_conversations()
    print(f"\n导出了{export_count}条对话记录")
    
    # 分析话题
    topics_df = analyzer.analyze_topics()
    print("\n热门话题:")
    print(topics_df.head())
    
    # 生成图表
    analyzer.plot_memory_growth()
    print("\n已生成记忆增长图表: memory_growth.png")