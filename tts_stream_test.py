# test_pcm_format.py
import numpy as np
import struct

def analyze_pcm_data(pcm_data: bytes, assumed_bit_depth: int = 16, assumed_channels: int = 1):
    """分析PCM数据的格式"""
    print(f"📊 分析PCM数据，大小: {len(pcm_data)} 字节")
    
    # 尝试不同格式解析
    formats_to_try = [
        ('int16', 2, np.int16),
        ('int32', 4, np.int32),
        ('float32', 4, np.float32),
        ('int8', 1, np.int8),
    ]
    
    for fmt_name, bytes_per_sample, dtype in formats_to_try:
        try:
            # 检查数据大小是否能被整除
            if len(pcm_data) % bytes_per_sample != 0:
                continue
                
            # 尝试解析
            array = np.frombuffer(pcm_data, dtype=dtype)
            
            # 计算统计数据
            min_val = np.min(array)
            max_val = np.max(array)
            mean_val = np.mean(array)
            std_val = np.std(array)
            
            # 判断是否合理
            if fmt_name == 'int16':
                valid_range = (-32768, 32767)
            elif fmt_name == 'int32':
                valid_range = (-2147483648, 2147483647)
            elif fmt_name == 'float32':
                valid_range = (-1.0, 1.0)
            elif fmt_name == 'int8':
                valid_range = (-128, 127)
            else:
                valid_range = (None, None)
            
            in_range = True
            if valid_range[0] is not None:
                in_range = (min_val >= valid_range[0] * 0.9 and max_val <= valid_range[1] * 0.9)
            
            print(f"  {fmt_name}: {len(array)}个样本, "
                  f"范围=[{min_val:.2f}, {max_val:.2f}], "
                  f"均值={mean_val:.2f}, 标准差={std_val:.2f}")
            
            if in_range:
                print(f"    ✅ 看起来像是{fmt_name}格式")
                return fmt_name, bytes_per_sample
            
        except Exception as e:
            print(f"  {fmt_name}: 解析失败 - {e}")
    
    print("❌ 无法确定PCM格式")
    return None, None

# 测试数据
test_data = b'\x00\x00\x10\x00\x20\x00\x30\x00'  # 示例16位PCM数据

print("测试PCM格式分析...")
fmt, bytes_per_sample = analyze_pcm_data(test_data)

if fmt:
    print(f"✅ 检测到的格式: {fmt}, 每样本{bytes_per_sample}字节")
else:
    print("❌ 未检测到有效格式")