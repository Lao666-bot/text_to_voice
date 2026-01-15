# 新增：topic_manager.py - 话题管理模块
import random
import time

class TopicManager:
    """话题管理器：主动话题、话题延续、话题切换"""
    
    def __init__(self):
        self.conversation_history = []
        self.topic_pool = {
            "日常": ["天气", "食物", "音乐", "电影", "书籍", "运动", "旅行"],
            "兴趣": ["爱好", "技能", "学习", "游戏", "艺术", "科技"],
            "人生": ["梦想", "目标", "回忆", "家庭", "朋友", "成长"],
            "趣味": ["假设问题", "脑筋急转弯", "分享趣事", "冷知识"]
        }
        self.current_topic = None
        self.topic_start_time = None
        self.topic_duration = 0
        
    def get_active_topic(self, idle_time: float) -> str:
        """根据空闲时间获取主动话题"""
        if idle_time > 60:  # 1分钟以上无交互
            return "好久没说话了，最近怎么样？"
        elif idle_time > 30:  # 30秒以上
            category = random.choice(list(self.topic_pool.keys()))
            topic = random.choice(self.topic_pool[category])
            
            # 根据话题类型生成不同的开场
            if category == "趣味":
                if topic == "假设问题":
                    questions = [
                        "如果有一天你能和动物说话，你会先和哪种动物聊天？",
                        "如果你能去任何一个时代生活一天，你会选择什么时候？",
                        "如果你有一种超能力，但只能在周二使用，你会选什么能力？"
                    ]
                    return random.choice(questions)
                elif topic == "分享趣事":
                    return "诶，我最近遇到一件很有趣的事情，你要不要听听看？"
            
            return f"对了，你平时喜欢{topic}吗？"
        
        return None  # 不需要主动话题
    
    def should_switch_topic(self) -> bool:
        """检查是否需要切换话题"""
        if not self.current_topic:
            return True
        
        # 话题持续时间过长（超过3分钟）
        if time.time() - self.topic_start_time > 180:
            return True
        
        # 最近几轮对话很简短
        if len(self.conversation_history) > 3:
            recent_lengths = [len(msg) for msg in self.conversation_history[-3:]]
            if max(recent_lengths) < 10:  # 最近3轮都很简短
                return True
        
        return False
    
    def record_conversation(self, user_input: str, ai_response: str):
        """记录对话"""
        self.conversation_history.append(user_input)
        self.conversation_history.append(ai_response)
        
        # 保持历史长度
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
    
    def get_topic_transition(self) -> str:
        """获取话题过渡语句"""
        transitions = [
            "对了，突然想到...",
            "话说回来...",
            "换个话题聊聊？",
            "诶，你知道吗...",
            "说起来..."
        ]
        return random.choice(transitions)