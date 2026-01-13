import psutil
import gc
import threading
import time
from typing import Optional
import random
class MemoryManager:
    def __init__(self, warning_threshold_mb: int = 4096, critical_threshold_mb: int = 6144):
        # 提高阈值，避免频繁触发
        self.warning_threshold = warning_threshold_mb * 1024 * 1024
        self.critical_threshold = critical_threshold_mb * 1024 * 1024
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.last_cleanup_time = 0
        self.cleanup_cooldown = 60  # 清理冷却时间60秒
    
    def monitor_memory(self, interval: float = 30.0):  # 增加监控间隔
        """监控内存使用情况，避免过于频繁"""
        while self.monitoring:
            usage = self.get_memory_usage()
            current_time = time.time()
            
            # 只在日志中显示，不频繁打印
            if random.random() < 0.1:  # 10%概率打印，减少日志
                print(f"📊 内存使用: {usage['rss_mb']:.1f} MB ({usage['percent']:.1f}%)")
            
            # 根据阈值采取不同行动
            if usage['rss_mb'] * 1024 * 1024 > self.critical_threshold:
                print(f"🚨 内存使用超过临界阈值({self.critical_threshold/1024/1024}MB)，执行紧急清理...")
                self.force_gc()
                self.clear_caches()
                self.last_cleanup_time = current_time
            elif usage['rss_mb'] * 1024 * 1024 > self.warning_threshold:
                # 检查冷却时间
                if current_time - self.last_cleanup_time > self.cleanup_cooldown:
                    print(f"⚠️  内存使用超过警告阈值({self.warning_threshold/1024/1024}MB)，执行清理...")
                    self.force_gc()
                    self.clear_caches()
                    self.last_cleanup_time = current_time
            
            time.sleep(interval)