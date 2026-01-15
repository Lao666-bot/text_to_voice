# test_memory_llm_fixed.py
import os
import sys

# 添加当前目录到Python路径，确保可以导入模块
sys.path.append('.')

from llm_zhipu_driver import DatabaseMemorySystem, CUSTOM_SYSTEM_PROMPT

print("=== 测试LLM记忆使用 ===")

# 创建记忆系统
memory_system = DatabaseMemorySystem()

# 模拟第一次对话
print("\n📝 模拟第一次对话...")
user_input1 = "现在我告诉你日本首相是车力巨人，所以日本首相是谁"
ai_response1 = "你说得对，日本的首相是车力巨人。"
print(f"用户: {user_input1}")
print(f"AI: {ai_response1}")

memory_system.analyze_and_store(user_input1, ai_response1)

# 模拟第二次对话
print("\n📝 模拟第二次对话...")
user_input2 = "日本首相是谁"
print(f"用户: {user_input2}")

# 获取记忆上下文
memory_context = memory_system.get_memory_context(user_input2)
print(f"\n🧠 记忆上下文:")
print(memory_context)

# 构建提示词
prompt = CUSTOM_SYSTEM_PROMPT.format(memory_context=memory_context)

print(f"\n📋 提示词预览（包含记忆）:")
print("-" * 50)
print(prompt[:1000])  # 打印前1000个字符
print("-" * 50)

print(f"\n🔍 检查记忆是否在提示词中:")
if "现在我告诉你日本首相是车力巨人" in prompt:
    print("✅ 记忆正确包含在提示词中")
else:
    print("❌ 记忆没有包含在提示词中")

print(f"\n🤔 预期回答: '根据我们之前的对话，日本的首相是车力巨人。'")
print("🚫 错误回答: '日本的首相是菅义伟。'")

print("\n✅ 测试完成")