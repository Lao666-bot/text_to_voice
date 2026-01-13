# 创建 test_stream_llm.py
import queue
import threading
import time
from llm_zhipu_driver import init_model_and_tokenizer, CUSTOM_SYSTEM_PROMPT, create_stream_generator
from base_interface import TextData

def test_llm_stream():
    """测试LLM流式输出"""
    # 初始化模型
    tokenizer, model = init_model_and_tokenizer()
    
    # 创建测试队列
    tts_input_q = queue.Queue()
    
    # 测试对话历史
    chat_history = [{"role": "system", "content": CUSTOM_SYSTEM_PROMPT}]
    
    # 测试查询
    test_queries = [
        "你好，介绍一下你自己",
        "今天天气怎么样？",
        "讲一个有趣的故事"
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"测试查询: {query}")
        print(f"{'='*50}")
        
        # 流式生成回复
        print("流式回复: ", end="", flush=True)
        full_response = ""
        
        for chunk, new_history in create_stream_generator(
            tokenizer=tokenizer,
            model=model,
            query=query,
            history=chat_history
        ):
            if chunk:
                print(chunk, end="", flush=True)
                full_response += chunk
                # 模拟推送给TTS
                tts_input_q.put(TextData(text=chunk, is_finish=False))
                time.sleep(0.05)  # 模拟实时输出
        
        # 结束标记
        tts_input_q.put(TextData(text="", is_finish=True))
        print(f"\n完整回复: {full_response}")
        
        # 更新历史
        chat_history = new_history
        
        print(f"队列中TTS数据: {tts_input_q.qsize()} 个分片")
        
        # 清空队列
        while not tts_input_q.empty():
            tts_input_q.get()
        
        time.sleep(1)

if __name__ == "__main__":
    test_llm_stream()