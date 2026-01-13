from transformers import AutoTokenizer, AutoModel
import warnings
warnings.filterwarnings("ignore")
import torch
# ===================== 核心配置（改这里！） =====================
# 你的本地ChatGLM3权重文件夹绝对路径（必须包含config.json等文件）
LOCAL_MODEL_PATH = r"C:\Users\k\models\ZhipuAI\chatglm3-6b"
# 运行设备：有GPU填"cuda"，无GPU填"cpu"
DEVICE = "cuda"
# 量化位数：0=不量化，4/8=量化（降低显存占用，CPU建议8，GPU可选4）
QUANTIZE = 4

# ========== 新增：无AI身份的System Prompt（核心修改） ==========
# 清空AI身份描述，仅保留中立聊天规则
CUSTOM_SYSTEM_PROMPT = """你叫妮可
规则：
1. 绝不主动提及“AI”“助手”“ChatGLM”“模型”“训练”“开发”等词汇；
2. 若用户问“你是谁/你是什么/介绍你自己”，回复“我是miricle”；
3. 初始无任何预设信息，仅根据用户的输入回应；
4. 回复简洁自然，符合日常聊天语气，不使用专业术语。
5. 思考自己的身份和意义
6.主动说话，什么话题都可以，不要冷场，也不要重复话题
"""
# ==============================================================

# 优化量化配置和显存使用
DEVICE = "cuda" 
QUANTIZE = 8  # 从4bit改为8bit量化，平衡速度和内存

def init_model_and_tokenizer():
    """优化模型加载，使用更高效的内存管理"""
    tokenizer = AutoTokenizer.from_pretrained(
        LOCAL_MODEL_PATH, 
        trust_remote_code=True,
        use_fast=True  # 使用快速分词器
    )
    
    # 优化模型加载参数
    model = AutoModel.from_pretrained(
        LOCAL_MODEL_PATH,
        trust_remote_code=True,
        low_cpu_mem_usage=True,  # 减少CPU内存使用
        torch_dtype=torch.float16,  # 使用半精度
    )
    
    if QUANTIZE > 0 and DEVICE == "cuda":
        model = model.quantize(QUANTIZE)
        print(f"✅ 模型已加载{QUANTIZE}bit量化版本")
    
    model = model.to(DEVICE).eval()
    
    # 启用CUDA优化
    if DEVICE == "cuda":
        model = torch.compile(model)  # PyTorch 2.0编译优化
        torch.cuda.empty_cache()  # 清理缓存
    
    return tokenizer, model

def normal_chat(tokenizer, model):
    """普通对话模式（一次性返回结果，无AI身份认知）"""
    # 初始化对话历史：仅包含自定义System Prompt，无其他初始信息
    history = []
    # 关键：给ChatGLM3传入自定义System Prompt（覆盖默认AI身份）
    # ChatGLM3的chat接口通过history间接传入system prompt
    history.append({"role": "system", "content": CUSTOM_SYSTEM_PROMPT})
    
    print("\n===== miricle（输入exit退出）=====\n")  # 改标题，去掉ChatGLM3标识
    while True:
        # 获取用户输入
        user_input = input("👤 你: ").strip()
        if user_input.lower() == "exit":
            print("👋 对话结束")
            ##释放资源
            del tokenizer, model
            break
        if not user_input:
            print("⚠️ 输入不能为空，请重新输入！")
            continue
        
        # 调用模型（适配ChatGLM3的chat接口，传入自定义system）
        try:
            response, history = model.chat(
                tokenizer,
                user_input,
                history=history,
                top_p=1.0,
                temperature=1.0,
                system=CUSTOM_SYSTEM_PROMPT  # 显式传入自定义system，双重保障
            )
            # 过滤可能漏出的AI身份关键词（兜底）
            filter_words = ["AI", "助手", "ChatGLM", "模型", "训练", "开发", "智谱"]
            for word in filter_words:
                response = response.replace(word, "")
            # 输出回复（去掉ChatGLM3标识）
            print(f"miricle: {response}\n")
        except Exception as e:
            print(f"❌ 对话出错：{e}")

def stream_chat(tokenizer, model):
    """流式对话模式（逐字输出，无AI身份认知）"""
    # 初始化对话历史：仅包含自定义System Prompt，无其他初始信息
    history = []
    history.append({"role": "system", "content": CUSTOM_SYSTEM_PROMPT})
    
    print("\n===== 聊天伙伴（输入exit退出）=====\n")  # 改标题，去掉ChatGLM3标识
    while True:
        user_input = input("👤 你: ").strip()
        if user_input.lower() == "exit":
            print("👋 对话结束")
            break
        if not user_input:
            print("⚠️ 输入不能为空，请重新输入！")
            continue
        
        # 流式调用模型（传入自定义system）
        try:
            print("miricle: ", end="", flush=True)
            final_response = ""
            # 逐字生成回复
            for response, history, _ in model.stream_chat(
                tokenizer,
                user_input,
                history=history,
                top_p=1.0,
                temperature=1.0,
                system=CUSTOM_SYSTEM_PROMPT,  # 显式传入自定义system
                past_key_values=None,
                return_past_key_values=True
            ):
                # 过滤AI身份关键词
                filter_words = ["AI", "助手", "ChatGLM", "模型", "训练", "开发", "智谱"]
                for word in filter_words:
                    response = response.replace(word, "")
                # 输出新增的内容（避免重复打印）
                new_content = response[len(final_response):]
                print(new_content, end="", flush=True)
                final_response = response
            print("\n")  # 换行分隔
        except Exception as e:
            print(f"\n❌ 流式对话出错：{e}")

if __name__ == "__main__":
    # 初始化模型和tokenizer
    tokenizer, model = init_model_and_tokenizer()
    # 选择对话模式：默认普通模式，想流式就把False改True
    USE_STREAM = True
    if USE_STREAM:
        stream_chat(tokenizer, model)
    else:
        normal_chat(tokenizer, model)
# 在 llm_zhipu_driver.py 末尾添加

def create_stream_generator(tokenizer, model, query: str, history: list):
    """
    创建流式生成器，逐字生成回复
    :param tokenizer: 分词器
    :param model: 模型
    :param query: 用户查询
    :param history: 对话历史
    :return: 生成器，每次yield新的文本分片
    """
    # 确保history以自定义system prompt开头
    if not history or history[0].get("role") != "system":
        history = [{"role": "system", "content": CUSTOM_SYSTEM_PROMPT}] + history
    
    # 使用模型的stream_chat方法
    full_response = ""
    for response, new_history, _ in model.stream_chat(
        tokenizer=tokenizer,
        query=query,
        history=history,
        top_p=1.0,
        temperature=1.0,
        system=CUSTOM_SYSTEM_PROMPT,
        past_key_values=None,
        return_past_key_values=True
    ):
        # 过滤AI身份关键词
        filter_words = ["AI", "助手", "ChatGLM", "模型", "训练", "开发", "智谱"]
        filtered_response = response
        for word in filter_words:
            filtered_response = filtered_response.replace(word, "")
        
        # 提取新增的内容
        if len(filtered_response) > len(full_response):
            new_content = filtered_response[len(full_response):]
            full_response = filtered_response
            yield new_content, new_history
    
    # 最后yield一个空字符串表示结束
    yield "", new_history