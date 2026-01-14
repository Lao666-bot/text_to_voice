#!/usr/bin/env python3
"""
ASR识别准确性诊断工具
用于测试和定位语音识别不准确的问题
"""
import os
import sys
import time
import queue
import threading
import numpy as np
import sounddevice as sd
import soundfile as sf
from pathlib import Path
from datetime import datetime

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入您的ASR模块
try:
    from funasr_driver import FunASRStreamingASR, AudioData, TextData
    from audio_player import AudioDriver
    print("✅ 模块导入成功")
except ImportError as e:
    print(f"❌ 模块导入失败: {e}")
    print("请确保以下模块已安装:")
    print("  pip install sounddevice soundfile")
    sys.exit(1)

# ===================== 测试配置 =====================
TEST_DURATION = 5  # 录音时长（秒）
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.6  # 分片时长
TEST_PHRASES = [
    "今天天气真好",
    "人工智能正在改变世界",
    "北京是中国的首都",
    "我想喝一杯咖啡",
    "明天下午三点开会",
    "这个系统运行得很流畅",
    "测试语音识别准确性",
    "欢迎使用智能语音助手"
]

# ===================== 测试1：直接录音并识别 =====================
def test_direct_recognition():
    """测试1：直接录音并识别（最接近真实使用场景）"""
    print("\n" + "="*60)
    print("测试1：直接录音并识别")
    print("="*60)
    
    # 初始化ASR
    print("🔄 初始化ASR模块...")
    asr_module = FunASRStreamingASR()
    
    # 提示用户说话
    print(f"🎤 请对着麦克风说一句话（{TEST_DURATION}秒）...")
    print("3秒后开始录音...")
    time.sleep(3)
    
    try:
        # 直接使用sounddevice录音（避免复杂队列）
        print("🔴 开始录音...")
        audio_data = sd.rec(
            int(TEST_DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype=np.int16
        )
        sd.wait()  # 等待录音完成
        print("✅ 录音完成")
        
        # 转换为AudioData格式
        pcm_bytes = audio_data.tobytes()
        audio_chunk = AudioData(
            pcm_data=pcm_bytes,
            sample_rate=SAMPLE_RATE,
            channels=1,
            is_finish=True
        )
        
        # 保存录音以便检查
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        wav_file = f"test_recording_{timestamp}.wav"
        sf.write(wav_file, audio_data, SAMPLE_RATE)
        print(f"💾 录音已保存: {wav_file}")
        
        # 使用ASR识别
        print("🔄 正在识别...")
        start_time = time.time()
        result = asr_module.process(audio_chunk)
        elapsed = time.time() - start_time
        
        print(f"⏱️  识别耗时: {elapsed:.2f}秒")
        print(f"📝 识别结果: {result.text}")
        
        return result.text, wav_file
        
    except Exception as e:
        print(f"❌ 测试1失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# ===================== 测试2：流式识别测试 =====================
def test_streaming_recognition():
    """测试2：模拟流式识别"""
    print("\n" + "="*60)
    print("测试2：流式识别测试")
    print("="*60)
    
    # 初始化ASR
    asr_module = FunASRStreamingASR()
    
    # 创建队列
    input_queue = queue.Queue()
    output_queue = queue.Queue()
    
    # 启动流式处理线程
    def run_asr():
        asr_module.stream_process(input_queue, output_queue)
    
    asr_thread = threading.Thread(target=run_asr, daemon=True)
    asr_thread.start()
    
    print("🎤 请对着麦克风说一句话（5秒）...")
    print("3秒后开始录音...")
    time.sleep(3)
    
    try:
        print("🔴 开始录音...")
        # 录制音频
        audio_data = sd.rec(
            int(5 * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype=np.int16
        )
        sd.wait()
        print("✅ 录音完成")
        
        # 将音频分成小块模拟流式输入
        chunk_samples = int(SAMPLE_RATE * CHUNK_DURATION)
        total_samples = len(audio_data)
        
        print(f"📊 音频总长度: {total_samples}样本，分片大小: {chunk_samples}样本")
        
        results = []
        for i in range(0, total_samples, chunk_samples):
            chunk = audio_data[i:min(i+chunk_samples, total_samples)]
            
            # 转换为AudioData
            audio_chunk = AudioData(
                pcm_data=chunk.tobytes(),
                sample_rate=SAMPLE_RATE,
                channels=1,
                is_finish=(i + chunk_samples >= total_samples)
            )
            
            # 推送到输入队列
            input_queue.put(audio_chunk)
            
            # 获取输出
            try:
                output = output_queue.get(timeout=0.5)
                if output.text:
                    results.append(output.text)
                    print(f"🔤 分片识别: {output.text}")
            except queue.Empty:
                pass
        
        # 发送结束标记
        input_queue.put(AudioData(pcm_data=b"", is_finish=True))
        
        # 等待最后的结果
        time.sleep(1)
        final_result = " ".join(results)
        
        print(f"📝 流式识别结果: {final_result}")
        
        return final_result
        
    except Exception as e:
        print(f"❌ 测试2失败: {e}")
        import traceback
        traceback.print_exc()
        return None

# ===================== 测试3：预录制音频测试 =====================
def test_pre_recorded_audio():
    """测试3：使用预录制音频文件测试"""
    print("\n" + "="*60)
    print("测试3：预录制音频测试")
    print("="*60)
    
    # 检查是否有预录制的测试文件
    test_files = [
        "test_audio.wav",
        "example.wav",
        "asr_example.wav"
    ]
    
    found_files = []
    for file in test_files:
        if os.path.exists(file):
            found_files.append(file)
    
    if not found_files:
        print("⚠️  未找到预录制的测试音频文件")
        print("请将测试音频文件（WAV格式，16kHz，单声道）放在当前目录")
        return None
    
    print(f"📁 找到测试文件: {found_files}")
    
    asr_module = FunASRStreamingASR()
    
    for audio_file in found_files[:2]:  # 测试前两个文件
        try:
            print(f"\n🔍 测试文件: {audio_file}")
            
            # 读取音频文件
            audio_data, sr = sf.read(audio_file)
            
            # 转换格式
            if sr != SAMPLE_RATE:
                print(f"⚠️  采样率不匹配: {sr}Hz -> {SAMPLE_RATE}Hz，正在转换...")
                # 简单重采样（实际应该用librosa或scipy）
                ratio = SAMPLE_RATE / sr
                new_length = int(len(audio_data) * ratio)
                indices = np.linspace(0, len(audio_data)-1, new_length).astype(int)
                audio_data = audio_data[indices]
            
            # 确保是单声道
            if len(audio_data.shape) > 1:
                audio_data = audio_data[:, 0]
            
            # 转换为16位整数
            audio_data = (audio_data * 32767).astype(np.int16)
            
            # 创建AudioData
            audio_chunk = AudioData(
                pcm_data=audio_data.tobytes(),
                sample_rate=SAMPLE_RATE,
                channels=1,
                is_finish=True
            )
            
            # 识别
            print("🔄 正在识别...")
            start_time = time.time()
            result = asr_module.process(audio_chunk)
            elapsed = time.time() - start_time
            
            print(f"⏱️  识别耗时: {elapsed:.2f}秒")
            print(f"📝 识别结果: {result.text}")
            
            # 播放音频供对比
            print("🔊 播放音频...")
            sd.play(audio_data, SAMPLE_RATE)
            sd.wait()
            
        except Exception as e:
            print(f"❌ 测试文件 {audio_file} 失败: {e}")
            continue
    
    return "测试完成"

# ===================== 测试4：不同模型对比 =====================
def test_different_models():
    """测试4：尝试不同的ASR模型"""
    print("\n" + "="*60)
    print("测试4：不同模型对比测试")
    print("="*60)
    
    print("⚠️  这个测试需要安装更多模型，可能耗时较长")
    print("是否继续？(y/n): ", end="")
    choice = input().strip().lower()
    
    if choice != 'y':
        print("跳过模型对比测试")
        return None
    
    models_to_test = [
        ("paraformer-zh-streaming", "v2.0.4"),
        ("paraformer-zh", "v2.0.4"),
        ("iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch", "v2.0.4"),
    ]
    
    results = {}
    
    # 先录制一段测试音频
    print("\n🎤 请说一句测试语句（3秒）...")
    time.sleep(2)
    
    print("🔴 开始录音...")
    test_audio = sd.rec(
        int(3 * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype=np.int16
    )
    sd.wait()
    print("✅ 录音完成")
    
    # 保存测试音频
    test_pcm = test_audio.tobytes()
    test_audio_data = AudioData(
        pcm_data=test_pcm,
        sample_rate=SAMPLE_RATE,
        channels=1,
        is_finish=True
    )
    
    # 播放给用户听
    print("🔊 播放您刚才的录音...")
    sd.play(test_audio, SAMPLE_RATE)
    sd.wait()
    
    print("您刚才说的是: ", end="")
    user_input = input().strip()
    
    for model_name, model_revision in models_to_test:
        try:
            print(f"\n🔄 测试模型: {model_name}")
            
            # 创建临时ASR实例
            from funasr import AutoModel
            test_model = AutoModel(
                model=model_name,
                model_revision=model_revision,
                disable_update=True
            )
            
            # 使用原始方法处理
            speech = np.frombuffer(test_pcm, dtype=np.int16)
            speech = speech.astype(np.float32) / 32767.0
            
            chunk_size = [0, 10, 5]
            chunk_stride = chunk_size[1] * 960
            
            total_chunk_num = int((len(speech)-1)/chunk_stride + 1)
            final_text = ""
            cache = {}
            
            for i in range(total_chunk_num):
                speech_chunk = speech[i*chunk_stride:(i+1)*chunk_stride]
                is_final = i == total_chunk_num - 1
                
                res = test_model.generate(
                    input=speech_chunk,
                    cache=cache,
                    is_final=is_final,
                    chunk_size=chunk_size,
                    encoder_chunk_look_back=4,
                    decoder_chunk_look_back=1
                )
                
                chunk_text = res[0]["text"] if res and len(res) > 0 else ""
                final_text += chunk_text
            
            results[model_name] = final_text
            print(f"📝 识别结果: {final_text}")
            
        except Exception as e:
            print(f"❌ 模型 {model_name} 测试失败: {e}")
            results[model_name] = f"ERROR: {e}"
    
    print("\n" + "="*60)
    print("模型对比结果:")
    print("="*60)
    print(f"原始语句: {user_input}")
    for model, result in results.items():
        print(f"{model}: {result}")
    
    return results

# ===================== 测试5：音频质量检查 =====================
def test_audio_quality():
    """测试5：检查音频输入质量"""
    print("\n" + "="*60)
    print("测试5：音频质量检查")
    print("="*60)
    
    print("🔍 检查音频设备...")
    try:
        devices = sd.query_devices()
        print(f"找到 {len(devices)} 个音频设备")
        
        default_input = sd.default.device[0]
        default_output = sd.default.device[1]
        
        print(f"默认输入设备: {default_input} - {devices[default_input]['name']}")
        print(f"默认输出设备: {default_output} - {devices[default_output]['name']}")
        
        # 检查输入设备参数
        input_info = devices[default_input]
        print(f"输入设备参数:")
        print(f"  最大输入通道数: {input_info['max_input_channels']}")
        print(f"  默认采样率: {input_info['default_samplerate']}")
        
        # 测试录音质量
        print("\n🎤 正在测试录音质量...")
        test_duration = 2
        
        # 录音
        print("🔴 请保持安静2秒...")
        time.sleep(1)
        silence = sd.rec(
            int(test_duration * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype=np.int16
        )
        sd.wait()
        
        # 计算噪音水平
        silence_rms = np.sqrt(np.mean(np.square(silence.astype(np.float32) / 32767.0)))
        print(f"📊 环境噪音水平: {silence_rms:.6f}")
        
        if silence_rms > 0.01:
            print("⚠️  环境噪音较高，可能影响识别")
        else:
            print("✅ 环境噪音水平正常")
        
        print("\n🔴 请说一句话测试...")
        time.sleep(1)
        speech = sd.rec(
            int(test_duration * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype=np.int16
        )
        sd.wait()
        
        # 计算语音能量
        speech_rms = np.sqrt(np.mean(np.square(speech.astype(np.float32) / 32767.0)))
        print(f"📊 语音能量水平: {speech_rms:.6f}")
        
        if speech_rms < 0.02:
            print("⚠️  语音信号较弱，建议靠近麦克风或提高音量")
        else:
            print("✅ 语音信号强度正常")
        
        # 计算信噪比（粗略）
        if silence_rms > 0:
            snr = 20 * np.log10(speech_rms / silence_rms) if speech_rms > 0 else 0
            print(f"📊 信噪比(SNR): {snr:.2f} dB")
            
            if snr < 10:
                print("⚠️  信噪比较低，语音可能被噪音干扰")
            else:
                print("✅ 信噪比良好")
        
        return {
            "noise_level": silence_rms,
            "speech_level": speech_rms,
            "snr": snr if 'snr' in locals() else 0
        }
        
    except Exception as e:
        print(f"❌ 音频质量检查失败: {e}")
        import traceback
        traceback.print_exc()
        return None

# ===================== 主测试函数 =====================
def main():
    """主测试函数"""
    print("""
    ███████╗███████╗██████╗      █████╗ ███████╗██████╗ 
    ██╔════╝██╔════╝██╔══██╗    ██╔══██╗██╔════╝██╔══██╗
    ███████╗███████╗██████╔╝    ███████║███████╗██████╔╝
    ╚════██║╚════██║██╔══██╗    ██╔══██║╚════██║██╔══██╗
    ███████║███████║██║  ██║    ██║  ██║███████║██║  ██║
    ╚══════╝╚══════╝╚═╝  ╚═╝    ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝
    ASR识别准确性诊断工具
    """)
    
    print("请选择测试项目:")
    print("1. 直接录音识别测试")
    print("2. 流式识别测试")
    print("3. 预录制音频测试")
    print("4. 不同模型对比测试")
    print("5. 音频质量检查")
    print("6. 全部测试")
    print("0. 退出")
    
    try:
        choice = input("请选择 (0-6): ").strip()
        
        if choice == '1':
            test_direct_recognition()
        elif choice == '2':
            test_streaming_recognition()
        elif choice == '3':
            test_pre_recorded_audio()
        elif choice == '4':
            test_different_models()
        elif choice == '5':
            test_audio_quality()
        elif choice == '6':
            print("\n🚀 开始全部测试...")
            test_direct_recognition()
            test_streaming_recognition()
            test_pre_recorded_audio()
            test_audio_quality()
            print("\n✅ 全部测试完成")
        elif choice == '0':
            print("退出测试")
            return
        else:
            print("无效选择")
            
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

# ===================== 快速诊断函数 =====================
def quick_diagnosis():
    """快速诊断：运行关键测试"""
    print("\n🚀 快速诊断开始...")
    
    # 1. 检查音频质量
    print("\n[1/3] 检查音频质量...")
    quality = test_audio_quality()
    
    # 2. 直接录音测试
    print("\n[2/3] 直接录音识别测试...")
    result, wav_file = test_direct_recognition()
    
    # 3. 提示用户对比
    print("\n" + "="*60)
    print("诊断建议:")
    print("="*60)
    
    if quality:
        if quality.get("snr", 0) < 10:
            print("🔴 问题: 环境噪音太高")
            print("建议: 在安静环境中测试，使用定向麦克风")
        
        if quality.get("speech_level", 0) < 0.02:
            print("🔴 问题: 语音信号太弱")
            print("建议: 靠近麦克风说话，提高音量")
    
    if result:
        print(f"📝 您的识别结果: {result}")
        print("请对比实际说话内容，判断识别准确性")
        
        if wav_file:
            print(f"💾 录音文件: {wav_file}")
            print("可以播放此文件检查录音质量")
    
    print("\n🔧 下一步:")
    print("1. 如果识别完全错误：可能是模型问题，尝试测试4（不同模型）")
    print("2. 如果部分错误：可能是音频质量问题，尝试改善录音环境")
    print("3. 如果延迟高：可能是硬件或配置问题")

# ===================== 执行入口 =====================
if __name__ == "__main__":
    # 检查依赖
    try:
        import sounddevice
        import soundfile
        import numpy
        print("✅ 所有依赖已安装")
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("请安装: pip install sounddevice soundfile numpy")
        sys.exit(1)
    
    print("\n欢迎使用ASR识别准确性诊断工具")
    print("本工具将帮助您找出语音识别不准确的原因")
    
    print("\n快速诊断模式？(y/n): ", end="")
    quick = input().strip().lower()
    
    if quick == 'y':
        quick_diagnosis()
    else:
        main()
    
    print("\n👋 诊断完成，希望对您有帮助！")