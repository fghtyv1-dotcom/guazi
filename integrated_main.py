import numpy as np
import joblib
import threading
import sounddevice as sd
import scipy.signal
import time
import os
import sys
import usb.core
import usb.util
import sys
import serial
# 改用 TensorFlow SavedModel (原文件推理, 比 ONNX 数值精度高, 无滑动窗口边界效应)
import tensorflow as tf
try:
    import tensorflow_hub as hub
except ImportError:
    hub = None
from sklearn.metrics.pairwise import cosine_similarity

# ========== 2. 核心配置 ==========
CLASS_NAMES = ["gun", "Nipponia-nippon", "snake", "weasel", "background"]
PRIORITY_MAP = {"gun": 1, "snake": 2, "weasel": 3, "Nipponia-nippon": 4, "background": 99}

MODEL_PATH = "species_classifier.pkl"
SCALER_PATH = "scaler.pkl"  # 训练配套的标准化工具
EMOTION_DB_PATH = "emotion_database.npy"  # 朱鹮情感特征指纹库
# 7/3: 用脚本所在目录解析路径, 从任何 cwd 跑都能找到模型 (跟 _onnx.py 同样改法)
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_FALLBACK = "/home/sunrise/sound/new/yamnet_compile"

def _resolve(name):
    for directory in (_BASE_DIR, _FALLBACK, "/home/sunrise"):
        path = os.path.join(directory, name)
        if os.path.exists(path):
            return path
    return os.path.join(_BASE_DIR, name)

MODEL_PATH      = _resolve("species_classifier.pkl")
SCALER_PATH     = _resolve("scaler.pkl")
EMOTION_DB_PATH = _resolve("emotion_database.npy")
# TF 原文件路径 (3.1MB SavedModel, 整段一次推理)
YAMNET_PATH = "/home/sunrise/sound/yamnet/yamnet/archive"

RATE = 16000
CHUNK_SIZE = 6144  # 0.384秒单帧大小
CONF_THRESHOLD = 0.30    # 蛇最高 0.33, 0.30 刚好触发
ANGLE_THRESHOLD = 20
SILENCE_THRESHOLD = 0.005  # 能量门限
EMOTION_THRESHOLD = 0.65  # 情感余弦相似度阈值门限

# 物种→固定角度 fallback（DOA 不可用时使用，跟 audio_detector.py 一致）
SPECIES_ANGLE_MAP = {"gun": 90, "Nipponia-nippon": 0, "snake": 135, "weasel": -90}

# 可选：强制使用指定 ALSA 设备（避免 PulseAudio 占着 XVF3800）
# 用法：ALSA_DEVICE=1,0 sudo -E python3 integrated_main.py
if os.environ.get('ALSA_DEVICE'):
    try:
        card, dev = (int(x) for x in os.environ['ALSA_DEVICE'].split(','))
        sd.default.device = (card, dev)
        print(f"🔧 强制使用 ALSA 设备: ({card}, {dev})")
    except Exception as exc:
        print(f"⚠️ ALSA_DEVICE 解析失败: {exc}")
# 舵机控制（与 visual_track.py 一致）
UART_PORT    = "/dev/mega0"   # udev 固定节点 (CH340 1a86:7523), 与 visual_track.py 一致
UART_BAUD    = 115200
MAX_ANGLE    = 90
MIN_ANGLE    = -90
SMOOTH       = 0.3
CMD_COOLDOWN = 0.15

# === DOA 平滑全局 (处理跨 0/360 边界 + EMA 吸收噪声) ===
_doa_smooth = None         # EMA 平滑后的 DOA (0..360), 初始 None
_doa_angle_prev = None     # 上一次 DOA (用于跨 0/360 边界处理)
_DOA_EMA_ALPHA = 0.3       # 新值权重 (0.3 = 30% 新 + 70% 历史)


# ========== 3. XVF3800 驱动 (适配 Linux) ==========
class ReSpeakerXVF3800:
    def __init__(self):
        self.dev = self._init_usb()
        self.angle_buffer = []

    def _init_usb(self):
        # 6/17: 诊断写文件 (sudo use_pty 拦截 stdout, 文件 IO 不受影响)
        _log = []
        try:
            dev = usb.core.find(idVendor=0x2886, idProduct=0x001A)
            _log.append(f"usb.core.find(2886:001a) = {dev}")
            if dev:
                try:
                    _kda = dev.is_kernel_driver_active(0)
                    _log.append(f"is_kernel_driver_active(0) = {_kda}")
                except Exception as _e:
                    _log.append(f"is_kernel_driver_active EXC: {_e}")
                try:
                    res = dev.ctrl_transfer(0xC0, 0, 0x80 | 18, 20, 5, 200)
                    _log.append(f"ctrl_transfer OK: bytes={list(res)}")
                    if res and len(res) >= 3:
                        _doa = (res[1] + (res[2] << 8)) % 360
                        _log.append(f"DOA = {_doa}deg")
                        _log.append("OK: self.dev != None -> get_angle real")
                        with open("/tmp/xvf_init.txt", "w") as _f:
                            _f.write("\n".join(_log))
                        print("✅ XVF3800 声源定位可用")
                        return dev
                except Exception as _e:
                    _log.append(f"ctrl_transfer EXC: {_e}")
                _log.append("FAIL: self.dev = None -> get_angle always 90")
                with open("/tmp/xvf_init.txt", "w") as _f:
                    _f.write("\n".join(_log))
                print("⚠️ XVF3800 声源定位不可用（无权限或资源占用），业务将按物种映射")
                return None
            else:
                _log.append("FAIL: device not found")
                with open("/tmp/xvf_init.txt", "w") as _f:
                    _f.write("\n".join(_log))
        except Exception as e:
            _log.append(f"OUTER EXC: {e}")
            _log.append("FAIL: self.dev = None -> get_angle always 90")
            try:
                with open("/tmp/xvf_init.txt", "w") as _f:
                    _f.write("\n".join(_log))
            except: pass
            print(f"⚠️ XVF3800 查找失败: {e}")
        return None

    def get_angle(self):
        if not self.dev:
            return 90
        try:
            resid = 20
            cmdid = 0x80 | 18  # 寄存器指令 0x92
            length = 5

            response = self.dev.ctrl_transfer(
                usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
                0, cmdid, resid, length, 500
            )

            if response and len(response) >= 3:
                angle = response[1] + (response[2] << 8)
                angle_val = angle % 360

                self.angle_buffer.append(angle_val)
                if len(self.angle_buffer) > 5:
                    self.angle_buffer.pop(0)
                return int(np.median(self.angle_buffer))
        except Exception:
            pass
        return 90


# ========== 4. 混合动力识别与情感双效引擎 ==========
class IntegratedEngine:
    def __init__(self):
        # 1. 加载 YAMNet (TF SavedModel 原文件)
        try:
            self.yamnet = hub.load(YAMNET_PATH) if hub else tf.saved_model.load(YAMNET_PATH)
            print("✅ YAMNet (TF SavedModel 原文件) 加载成功")
        except Exception as e:
            sys.exit(f"❌ YAMNet 加载失败: {e}")

        # 2. 加载物种分类器
        self.species_clf = joblib.load(MODEL_PATH)

        # 3. 加载特征空间标准化器
        if os.path.exists(SCALER_PATH):
            self.scaler = joblib.load(SCALER_PATH)
            print("✅ 特征空间校准器 Scaler 加载成功")
        else:
            sys.exit(f"❌ 缺少 {SCALER_PATH} 文件，请先运行训练脚本！")

        # 4. ⭐️ 核心集成：轻量化加载朱鹮情感指纹库 (.npy 字典文件)
        if os.path.exists(EMOTION_DB_PATH):
            try:
                self.emotion_db = np.load(EMOTION_DB_PATH, allow_pickle=True).item()
                print(f"✅ 朱鹮情感指纹库加载成功 (包含类别: {list(self.emotion_db.keys())})")
            except Exception as e:
                self.emotion_db = None
                print(f"⚠️ 情感指纹库解析失败: {e}，已关闭情感分析功能")
        else:
            self.emotion_db = None
            print("⚠️ 未找到情感指纹库文件 emotion_database.npy，已跳过情感追踪")

    def analyze(self, audio_data):
        # 1. 硬件级带通滤波预处理
        nyq = 0.5 * RATE
        b, a = scipy.signal.butter(4, [200 / nyq, 7500 / nyq], btype='band')
        filtered = scipy.signal.lfilter(b, a, audio_data).astype(np.float32)

        # 2. 通过 YAMNet (TF SavedModel 原文件) 整段一次推理提取 embeddings
        #    输出 (scores, embeddings, log_mel_spectrogram) - embeddings 形状 (N, 1024)
        _, embeddings, _ = self.yamnet(filtered)
        if embeddings.shape[0] == 0:
            return None
        # === C. TF tensor 优化: 直接 reduce_mean 避免 .numpy().mean() 双拷贝 ===
        mean_emb = tf.reduce_mean(embeddings, axis=0).numpy().reshape(1, -1)

        # 3. 消除系统偏置 (StandardScaler)
        mean_emb_scaled = self.scaler.transform(mean_emb)

        # 4. 预测物种概率
        probs = self.species_clf.predict_proba(mean_emb_scaled)[0]
        idx = np.argmax(probs)
        detected_species = CLASS_NAMES[idx]
        prio = PRIORITY_MAP.get(detected_species, 99)

        # 5. ⭐️⭐️⭐️ 核心集成：联动的朱鹮情感匹配算法 ⭐️⭐️⭐️
        emotion_res = None
        max_sim = 0.0

        if detected_species == "Nipponia-nippon" and self.emotion_db:
            # 遍历指纹库中的每种情感基准向量进行余弦相似度碰撞
            for emo_name, base_v in self.emotion_db.items():
                # 确保维度对齐
                base_v_reshaped = base_v.reshape(1, -1)
                # 计算余弦相似度
                sim = cosine_similarity(mean_emb, base_v_reshaped)[0][0]

                if sim > max_sim:
                    max_sim = sim
                    if sim > EMOTION_THRESHOLD:
                        emotion_res = emo_name

        return {
            "species": detected_species,
            "conf": probs[idx],
            "emb": mean_emb,
            "prio": prio,
            "emotion": emotion_res,
            "emotion_sim": max_sim,
            "probs": probs     # 5 类全部分布
        }


# ========== 5. 视觉联动接口 ==========
def notify_vision_system(species, angle, emotion=None):
    emo_str = f" | 情感状态: {emotion}" if emotion else ""
    print(f"📡 [发送视觉信号] 锁定目标: {species} | 角度: {angle}°{emo_str}")


# ========== 6. 主循环 ==========
if __name__ == "__main__":
    hw = ReSpeakerXVF3800()
    engine = IntegratedEngine()
    last_angle = -1
    last_species_time = {}  # 物种冷却计时 (6/17, 10s)

    GUN_AMP_THRESHOLD = 0.15
    GUN_CONF_THRESHOLD = 0.50
    # 舵机初始化（与 visual_track.py 一致）
    motor = None
    current_motor_angle = 0
    last_cmd_time = 0
    motor_lock = threading.Lock()
    try:
        motor = serial.Serial(UART_PORT, UART_BAUD, timeout=0.3)
        print(f"✅ 舵机串口已打开: {UART_PORT} @ {UART_BAUD}")
    except Exception as e:
        print(f"⚠️ 舵机串口打开失败（{e}），云台转动将不可用")

    def set_motor_angle(target_deg):
        """限幅+平滑+冷却+串口发送（与视觉逻辑一致）"""
        global current_motor_angle, last_cmd_time
        if motor is None:
            return
        target = max(MIN_ANGLE, min(MAX_ANGLE, target_deg))
        with motor_lock:
            now = time.time()
            if now - last_cmd_time < CMD_COOLDOWN:
                return
            if abs(target - current_motor_angle) < 2:
                return
            # 6/17 取消 SMOOTH
            current_motor_angle = target  # 直接赋值
            current_motor_angle = max(MIN_ANGLE, min(MAX_ANGLE, current_motor_angle))
            try:
                motor.write(f"T{int(current_motor_angle)}\n".encode())
                last_cmd_time = now
            except Exception as exc:
                print(f"⚠️ 舵机串口写入失败: {exc}")


    print("\n[系统就绪] 开始在 Radxa Rock Pi 5X 上监测野生动物生态声音...")

    # === A. 异步录音流 + B. 预分配 buffer 复用 (A+B 组合, 消除 sd.rec/sd.wait 时序耦合) ===
    _audio_buf = np.zeros(int(2.0 * RATE), dtype=np.float32)  # B: 预分配 2 秒 buffer
    _audio_lock = threading.Lock()
    _prev_quiet = True            # 6/17: 安静→有声过渡检测
    _skip_analyze = 0             # 6/17: 过渡后跳过次数 (0.2s/次, 10=跳过 2 秒)
    def _audio_cb(indata, frames, time_info, status):
        """sounddevice 回调: 持续把新样本滚入 _audio_buf (in-place, 无新内存分配)"""
        global _audio_buf, _prev_quiet, _skip_analyze
        samples = indata[:, 0].astype(np.float32)
        n = len(samples)
        # 6/17: 检测安静→有声过渡, 重置 buffer 避免旧静音污染
        # (前 1-2 秒的"半静音+半声"被 YAMNet 误判成朱鹮, 这是原因)
        is_loud = float(np.max(np.abs(samples))) > 0.01
        if _prev_quiet and is_loud:
            with _audio_lock:
                _audio_buf[:] = 0  # 清空旧数据
            _skip_analyze = 10  # 跳过 10 * 0.2s = 2 秒, 等 buffer 填满真实数据
        _prev_quiet = not is_loud
        with _audio_lock:
            _audio_buf[:-n] = _audio_buf[n:]   # 移位 (in-place)
            _audio_buf[-n:] = samples           # 写入
    # 6/17: 显式 device=(1, 0), 不依赖 sd.default.device (避免被 PulseAudio pulse 设备抢走, 报 -9998)
    _device = (1, 0)
    print(f"🔊 打开 InputStream: device={_device}, channels=2, rate=16000")
    _audio_stream = sd.InputStream(
        device=_device,
        samplerate=RATE, channels=2, dtype='float32',  # 6/17: XVF3800 硬件是 2 声道, callback 取 [:, 0]
        blocksize=int(RATE * 0.1),  # 0.1 秒/块
        callback=_audio_cb
    )
    _audio_stream.start()
    print("✅ 异步录音流启动 (0.1s/块, 2s 滑动窗口)")

    try:
        # 6/17: 预热, 让 _audio_buf 填满真实数据 (避免初始 0 污染导致过渡误判)
        print("⏳ 预热 2 秒, 等 buffer 填满真实数据...")
        time.sleep(2.0)
        _skip_analyze = 0
        print("✅ 预热完成, 进入识别周期")
        while True:
            # 6/17: 过渡后跳过, 让 buffer 填满真实数据
            if _skip_analyze > 0:
                _skip_analyze -= 1
                time.sleep(0.2)
                continue
            # 轮询 buffer (200ms 一次, < 1秒延迟)
            time.sleep(0.2)
            with _audio_lock:
                audio_data = _audio_buf.copy()  # 防御性 copy (32KB, < 1ms)

            # 能量门限（环境安静时不进入周期，节省 CPU 算力）
            current_amp = np.max(np.abs(audio_data))
            if current_amp < SILENCE_THRESHOLD:
                continue

            # ⭐️ 核心修复 (6/16): RMS 归一化, 对齐离线 .m4a 文件尺度
            # 原因: 实时麦克风 max_amp 通常 0.05-0.2 (手机外放/距离/拾音弱),
            #       离线 .m4a 是 0.7-1.0. scaler.pkl 按训练数据 ~0.7 fit,
            #       实时 amp 0.05 喂进去会被 scaler 拉到训练分布外, conf 跌 0.10-0.20.
            # 验证 (test_normalization.py 6/16): ×0.1 模拟 amp=0.006 → weasel 0.65→0.26 (误判朱鹮),
            #       RMS 归一化后 → weasel 0.69 (修复). 峰值归一化 0.67 也行, RMS 略稳.
            # 用归一化前 amp 做门限, 避免放大噪声. SILENCE_THRESHOLD=0.04 防止纯噪声触发.
            _rms = float(np.sqrt(np.mean(audio_data ** 2)) + 1e-9)
            # 自适应增益: 极安静 (RMS<0.01) 时拉到 0.05, 否则归一化到 0.1
            if _rms < 0.01:
                audio_data = audio_data / _rms * 0.05
            else:
                audio_data = audio_data / _rms * 0.1

            # 开始推理
            result = engine.analyze(audio_data)

            if result:
                species = result['species']
                conf = result['conf']

                # 0. 通用置信度过滤 (修复: 之前 CONF_THRESHOLD 定义了但没生效)
                #    0.65 = 同学建议阈值, 防止低置信度背景音乱触发电机
                # 类别分通道阈值 (A 方案)
                #   原因: 麦克风录音下蛇/黄鼠狼/枪声 embeddings 信噪比差, 朱鹮训练样本多容易被误判
                #   朱鹮: 严 (>=0.50), 防止背景噪音被误判成朱鹮
                #   其他: 宽 (>=0.15), 让低置信度真实检测通过
                # 通用阈值: 0.10 (最宽松, 只过滤完全没声)
                # 用户自定义权威规则: 蛇/黄鼠狼 conf > 0.15 就强制识别成它
                #   (即使模型 argmax 报别的, 也以蛇/黄鼠狼为权威)
                #   CLASS_NAMES: gun=0, Nipponia-nippon=1, snake=2, weasel=3, background=4
                if 'probs' in result:
                    probs = result['probs']
                    if probs[2] > 0.30:        # snake (snake 实测 0.32-0.33, 阈值 0.30 刚好)
                        species = 'snake'
                        conf = probs[2]
                    elif probs[3] > 0.60:    # weasel (实测 0.41-0.75, 阈值 0.60 更保守)
                        species = 'weasel'
                        conf = probs[3]
                if conf < 0.10: continue
                # 1. 过滤环境背景音
                if species == "background":
                    continue

                # 1.5 振幅过滤 (6/17 保留: 防弱信号误判)
                if current_amp < GUN_AMP_THRESHOLD:
                    continue
                # 1.6 枪声置信度过滤 (6/17 新增: 之前 dead code, 实际没用上, 导致 conf=0.32 也触发)
                #   用户反馈 "枪声特别容易误判", 用 GUN_CONF_THRESHOLD 兜底
                if species == "gun" and conf < GUN_CONF_THRESHOLD:
                    continue

                # 2. 针对枪声防误报严苛过滤

                # 3. 获取声源方位角度
                if hw.dev is not None:
                    current_angle = hw.get_angle()
                else:
                    # DOA 不可用，按物种映射固定角度（跟 audio_detector.py 一致）
                    current_angle = SPECIES_ANGLE_MAP.get(species, 90)
                    print(f"⚠️ DOA 不可用，按物种 {species} 映射角度 {current_angle}°")

                # 3.5 DOA 平滑: 处理 0/360 边界 + EMA 吸收噪声 (避免 DOA 跳变让电机乱跳)
                # 注: if __name__ == "__main__" 块是模块级代码, 不需要 global 关键字
                if _doa_smooth is None:
                    _doa_smooth = float(current_angle)
                    _doa_angle_prev = float(current_angle)
                else:
                    diff = current_angle - _doa_angle_prev
                    if diff > 180:   diff -= 360   # 跨 0/360 边界: 选短边
                    elif diff < -180: diff += 360
                    _doa_smooth = (_doa_smooth + diff * _DOA_EMA_ALPHA) % 360
                    if _doa_smooth < 0: _doa_smooth += 360
                    _doa_angle_prev = float(current_angle)

                # 4. 触发判定：若角度位移，或识别到高优先级目标（枪声、朱鹮等），立即联动云台
                if abs(_doa_smooth - last_angle) > ANGLE_THRESHOLD or result['prio'] <= 4:
                    # 10 秒冷却: 同一物种冷却期内不重复触发 (6/17)
                    now = time.time()
                    _last = last_species_time.get(species, 0)
                    if now - _last < 10.0:
                        continue  # 6/17: 同一物种 10 秒冷却 (枪声也走, 防止误判)
                    last_species_time[species] = now
                    probs_str = " | ".join(f"{CLASS_NAMES[i]}:{p:.2f}" for i, p in enumerate(result['probs']))
                    print(f"\n🎯 捕获: {species} (conf={conf:.2f}) | 全部: [{probs_str}]")

                    # ⭐️ 打印朱鹮的情感状态
                    if species == "Nipponia-nippon":
                        if result['emotion']:
                            print(
                                f"💓 [状态预警] 检测到朱鹮处于: {result['emotion']} 状态 (指纹相似度: {result['emotion_sim']:.2f})")
                        else:
                            print(f"🐾 [状态常规] 检测到朱鹮，但特征相似度较低 (最高相似度: {result['emotion_sim']:.2f})")

                    print(f"🔄 指令发放 -> 云台紧急转动至: {int(current_angle)}° (DOA 原始)")
                    notify_vision_system(species, int(_doa_smooth), result['emotion'])
                    last_angle = _doa_smooth
                    # 6/17: 舵机改用 current_angle (原始 DOA), 不用 _doa_smooth
                    #   bug: _doa_smooth EMA 累积在某些情况下跑偏 (实测 _doa_smooth=254 但 current_angle=70)
                    #   修法: 触发判定仍用 _doa_smooth (保持平滑), 实际指向用 current_angle (精确)
                    motor_target = current_angle
                    if motor_target > 180: motor_target -= 360  # wrap 到 -180..180
                    motor_target = max(-90, min(90, motor_target))  # 限幅 -90..+90
                    set_motor_angle(motor_target)
                    print(f"🎮 舵机指令 -> T{int(current_motor_angle)} (DOA={int(current_angle)}°, 目标角={int(motor_target)}°)")



    except KeyboardInterrupt:
        print("\n👋 监测程序已被手动停止。")
