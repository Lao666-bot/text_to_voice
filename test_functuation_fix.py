#!/usr/bin/env python3
"""
标点模型问题诊断和修复测试
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from funasr_driver import FunASRStreamingASR

def test_punctuation_models():
    """测试不同的标点模型处理方式"""
    print("🧪 测试标点模型...")
    
    # 测试句子
    test_sentences = [
        "你觉得孙悟空打哪吒谁会赢",
        "今天天气真好",
        "这个系统运行得很流畅吗",
        "欢迎使用智能语音助手",
        "赢赢赢",  # 测试重复问题
        "哪吒。谁会赢",  # 测试已有标点的情况
    ]
    
    # 初始化ASR（只为了用标点模型）
    print("🔄 初始化ASR模块...")
    asr_module = FunASRStreamingASR()
    
    print("\n" + "="*60)
    print("测试标点模型处理")
    print("="*60)
    
    for sentence in test_sentences:
        print(f"\n📝 测试句子: '{sentence}'")
        
        # 测试当前标点模型
        if asr_module.use_punc_model and asr_module.punc_model is not None:
            try:
                print("🔤 调用标点模型...")
                result = asr_module.punc_model.generate(input=sentence)
                print(f"📊 模型输出: {result}")
                
                # 提取文本
                punctuated = asr_module._extract_text_from_punc_result(result)
                print(f"✅ 标点结果: '{punctuated}'")
            except Exception as e:
                print(f"❌ 模型处理失败: {e}")
        else:
            print("⚠️  标点模型未加载，使用规则标点")
            punctuated = asr_module._smart_rule_based_punc(sentence)
            print(f"✅ 规则标点: '{punctuated}'")

def test_streaming_simulation():
    """模拟流式处理，发现问题"""
    print("\n" + "="*60)
    print("模拟流式处理")
    print("="*60)
    
    # 模拟流式输入（分片）
    stream_chunks = [
        "你觉得孙",
        "悟空",
        "打哪吒",
        "谁会赢",
        "赢",  # 模拟重复
        "赢",
    ]
    
    print("模拟ASR流式输出:")
    for i, chunk in enumerate(stream_chunks):
        print(f"分片 {i+1}: '{chunk}'")
    
    # 模拟当前逻辑
    print("\n模拟当前缓存逻辑:")
    punc_buffer = ""
    for chunk in stream_chunks:
        punc_buffer += chunk
        print(f"缓存: '{punc_buffer}'")
        
        # 模拟_add_punctuation逻辑
        if len(punc_buffer) >= 5:  # 假设长度足够
            # 假设模型输出
            model_output = punc_buffer + "？"
            # 当前逻辑：保留最后2个字符
            output = model_output[:-2] if len(model_output) > 2 else ""
            punc_buffer = model_output[-2:] if len(model_output) > 2 else model_output
            print(f"  模型输入: '{punc_buffer}'")
            print(f"  模型输出: '{model_output}'")
            print(f"  实际输出: '{output}'")
            print(f"  新缓存: '{punc_buffer}'")

def test_fixed_logic():
    """测试修复后的逻辑"""
    print("\n" + "="*60)
    print("测试修复后的逻辑")
    print("="*60)
    
    # 简化的修复逻辑
    class FixedPunctuationLogic:
        def __init__(self):
            self.sentence_buffer = ""
            
        def process_chunk(self, chunk_text, is_final=False):
            """处理分片文本"""
            if not chunk_text:
                return ""
                
            self.sentence_buffer += chunk_text
            
            # 只在句子足够长或结束时处理
            if len(self.sentence_buffer) >= 8 or is_final:
                # 简单规则：疑问句加问号，其他加句号
                sentence = self.sentence_buffer
                
                # 检查是否是疑问句
                question_words = ['吗', '呢', '吧', '啊', '什么', '为什么', '怎么', '如何', '谁', '哪']
                is_question = any(word in sentence for word in question_words) or sentence.endswith(tuple(question_words))
                
                if is_question:
                    result = sentence + "？"
                else:
                    result = sentence + "。"
                    
                # 清空缓存
                self.sentence_buffer = ""
                return result
            else:
                return ""  # 不输出，继续累积
    
    # 测试
    logic = FixedPunctuationLogic()
    test_chunks = ["你觉得", "孙悟空", "打哪吒", "谁会", "赢"]
    
    print("流式处理:")
    full_sentence = ""
    for chunk in test_chunks:
        result = logic.process_chunk(chunk)
        if result:
            print(f"输出完整句子: '{result}'")
            full_sentence = result
    
    # 最后强制结束
    if logic.sentence_buffer:
        result = logic.process_chunk("", is_final=True)
        if result:
            print(f"最终输出: '{result}'")
            full_sentence = result
    
    print(f"\n最终结果: '{full_sentence}'")

def main():
    """主函数"""
    print("标点模型问题诊断工具")
    print("="*60)
    
    print("1. 测试标点模型基础功能")
    print("2. 模拟流式处理发现问题")
    print("3. 测试修复后的逻辑")
    print("4. 全部测试")
    
    choice = input("\n请选择测试项目 (1-4): ").strip()
    
    if choice == '1':
        test_punctuation_models()
    elif choice == '2':
        test_streaming_simulation()
    elif choice == '3':
        test_fixed_logic()
    elif choice == '4':
        test_punctuation_models()
        test_streaming_simulation()
        test_fixed_logic()
    else:
        print("无效选择")

if __name__ == "__main__":
    main()