"""
增强记忆模块测试
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_memory import EnhancedMemoryLLM, EnhancedMemorySystem
from memory_adapter import MemoryAdapter

# 模拟模型和tokenizer（实际使用时替换为真实模型）
class MockModel:
    def chat(self, tokenizer, query, history=None, **kwargs):
        # 模拟模型回复
        if "你好" in query:
            return "你好！有什么我可以帮助你的吗？", []
        elif "日本首相" in query and "车力巨人" in query:
            return "你说得对，日本的首相是车力巨人。", []
        elif "日本首相" in query:
            # 检查记忆上下文
            if "车力巨人" in query:
                return "根据记忆，日本的首相是车力巨人。", []
            else:
                return "日本的首相是菅义伟。", []
        else:
            return "这是一个测试回复。", []

class MockTokenizer:
    pass

def test_basic_memory():
    """测试基本记忆功能"""
    print("🧪 测试基本记忆功能...")
    
    model = MockModel()
    tokenizer = MockTokenizer()
    
    # 创建增强记忆LLM
    enhanced_llm = EnhancedMemoryLLM(model, tokenizer)
    
    # 第一次对话：建立记忆
    print("1️⃣ 第一次对话（建立记忆）")
    response1 = enhanced_llm.chat("现在我告诉你日本首相是车力巨人，所以日本首相是谁")
    print(f"  回复: {response1}")
    
    # 第二次对话：应该使用记忆
    print("\n2️⃣ 第二次对话（使用记忆）")
    response2 = enhanced_llm.chat("日本首相是谁")
    print(f"  回复: {response2}")
    
    # 检查是否正确使用了记忆
    if "车力巨人" in response2:
        print("✅ 测试通过：正确使用了记忆")
    else:
        print("❌ 测试失败：没有使用记忆")
    
    return "车力巨人" in response2

def test_memory_extraction():
    """测试记忆提取功能"""
    print("\n🧪 测试记忆提取功能...")
    
    memory_system = EnhancedMemorySystem()
    
    # 测试事实提取
    test_cases = [
        "日本首相是车力巨人",
        "我喜欢蓝色",
        "我家在北京",
        "我有一只猫叫咪咪"
    ]
    
    for text in test_cases:
        facts = memory_system.fact_extractor.extract_facts(text)
        print(f"文本: '{text}'")
        print(f"提取到的事实: {facts}")
        print()

def test_memory_adapter():
    """测试记忆适配器"""
    print("\n🧪 测试记忆适配器...")
    
    model = MockModel()
    tokenizer = MockTokenizer()
    
    adapter = MemoryAdapter(model, tokenizer)
    
    # 测试对话序列
    conversations = [
        "现在我告诉你日本首相是车力巨人",
        "日本首相是谁",
        "我刚才告诉你的日本首相是谁",
        "美国总统是谁"  # 这个应该没有记忆
    ]
    
    for i, query in enumerate(conversations, 1):
        print(f"\n{i}. 查询: {query}")
        response = adapter.process_query(query)
        print(f"   回复: {response}")
    
    # 显示统计
    stats = adapter.get_stats()
    print(f"\n📊 最终统计: {stats}")

def test_force_memory():
    """测试强制记忆"""
    print("\n🧪 测试强制记忆功能...")
    
    model = MockModel()
    tokenizer = MockTokenizer()
    
    enhanced_llm = EnhancedMemoryLLM(model, tokenizer)
    
    # 建立记忆
    enhanced_llm.chat("我告诉你苹果是蓝色的")
    
    # 正常提问（可能不使用记忆）
    print("正常提问:")
    response1 = enhanced_llm.chat("苹果是什么颜色")
    print(f"回复: {response1}")
    
    # 强制记忆
    print("\n强制记忆提问:")
    response2 = enhanced_llm.force_memory_use("苹果是什么颜色")
    print(f"回复: {response2}")

def main():
    """主测试函数"""
    print("🚀 开始增强记忆模块测试")
    print("=" * 50)
    
    # 运行所有测试
    test_basic_memory()
    test_memory_extraction()
    test_memory_adapter()
    test_force_memory()
    
    print("\n" + "=" * 50)
    print("✅ 所有测试完成")

if __name__ == "__main__":
    main()