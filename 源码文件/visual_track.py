def should_i_drive_servo():
    """6/19 dual-mode: 读 /tmp/tracking_mode.json 决定 visual 是否写舵机
    TRACK_MODE='video' -> visual 写, audio 跳过 (visual 写)
    TRACK_MODE='audio' -> audio 写, visual 跳过 (visual 不写)
    默认 'video' (visual 主导)
    """
    try:
        import json as _json_v
        with open('/tmp/tracking_mode.json') as _f_v:
            _mode = _json_v.load(_f_v).get('TRACK_MODE', 'video')
        return _mode == 'video'
    except Exception:
        return True
#!/usr/bin/env python3
"""visual_track.py — BPU 视觉推理 + 电机1追踪 + 网页推流"""
import cv2, numpy as np, time, threading, os
import sounddevice as sd
import sys
import json
sys.stdout.reconfigure(line_buffering=True)   # 让 nohup 日志能立刻看到 [mega] 调试输出
from flask import Flask, Response, request
from hobot_dnn import pyeasy_dnn as dnn
import serial
from collections import deque

_external_motor_handler = None
_external_event_handler = None
_external_audio_bins_provider = None

# mega 启动/调试日志 ring buffer（最近 80 行，供 /api/status 拉取）
mega_log = deque(maxlen=80)
# === 报警网格 (4x4 压力垫) 共享状态 ===
# 早期版本由 pressure_module 线程更新；现独立进程处理，本地占位即可（路由永远返回全 0 也不报错）
alarm_lock = threading.Lock()
alarm_grid  = [[False]*4 for _ in range(4)]

def find_mega_port():
    """按 CH340 (1a86:7523) 的 VID:PID 在 USB 总线上找 mega2560。
    解决 Linux 内核 ttyUSB* 编号漂移问题：哪怕用户配置的 /dev/ttyUSB0 不存在，
    也能用这个 fallback 找到实际设备。"""
    try:
        from serial.tools import list_ports
        for p in list_ports.comports():
            if p.vid == 0x1a86 and p.pid == 0x7523:
                return p.device
    except Exception as e:
        print(f"⚠️ 扫 CH340 失败: {e}", flush=True)
    return None

# ──── 配置 ────
MODEL_PATH   = "/home/sunrise/test/best_aivideo.bin"
CAMERA_ID    = 0
CONF_THRESH  = 0.25
STREAM_PORT  = 5000
UART_PORT    = "/dev/ttyUSB0"
UART_BAUD    = 115200
FILE_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_PATH = os.path.join(FILE_BASE_DIR, "dashboard_final.html")
LEGACY_DASHBOARD_PATH = os.path.join(FILE_BASE_DIR, "feiyu (2).html")
PREFER_EMBEDDED_DASHBOARD = False

CLS_NAMES    = ['crested_ibis', 'hunter', 'other_birds', 'weasel', 'snake']

# ── 运行时可调参数（通过 /control 面板修改，无需重启）──
# === cap_thread 共享 frame (v6 风格多线程) ===
shared_lock = threading.Lock()
shared_frame = None
frame_seq = 0
frame_ready = threading.Event()

PARAMS = {
    'TRACK_CLASS':  None,   # None=跟踪置信度最高的任意类，或填 'hunter' 等
    'DEADZONE':     30,     # 像素死区（提高精度条，目标在 ±30 像素内电机不响应，否则开始响应）
    'MAX_ANGLE':    90,     # 最大角度
    'MIN_ANGLE':   -90,
    'STEP_RATIO':   0.25,   # 每次最多走剩余量的 25%
    'CMD_COOLDOWN': 0.30,   # 指令最小间隔(秒) 抬高
    'ANGLE_EPS':    5,      # 角度死区
    'CONF_THRESH':  0.5,    # BPU 推理 mask 阈值（YOLOv5 标准默认；可 /api/set CONF_THRESH 实时调）
    'TRACK_ENABLED': True,  # 总开关：开跟踪才会下发电机指令
    'IDLE_HOME_ENABLED': True,   # 无目标一段时间后自动回 0°
    'IDLE_HOME_TIMEOUT': 5.0,    # 无目标多少秒后回 0°
    'STABLE_FRAMES': 3,     # 防抖：连续 N 帧同向才下发
    'UART_PORT':    '/dev/mega0',   # udev 规则固定的 CH340 节点（按 1a86:7523）   # udev 规则固定的 CH340 节点（按 1a86:7523）
    'UART_BAUD':    115200,
    'KF_Q_POS':     2.0,     # KF 过程噪声方差 - 位置分量  (越大越激进跟随)
    'KF_Q_VEL':     8.0,     # KF 过程噪声方差 - 速度分量  (越大越激进跟随速度变化)
    'KF_R_MEAS':    144.0,   # KF 测量噪声方差 (= 12^2; BPU 量化噪声 std≈12 px 经验值)
    'KF_V_DECAY':    0.95,   # v 泄漏衰减: 1.0=不衰减(当前), 0.95=每帧衰减5%防止过冲
    'BOX_SCALE':     1.15,    # 画框经验系数: 1.0=raw(BPU 框偏小), 1.15≈真值; 调到 1.2/1.25 看效果
    'TRACK_MODE':   'video',  # 'video' | 'audio': 切换视觉追踪/声纹追踪
}
MIN_ANGLE     = PARAMS['MIN_ANGLE']
MAX_ANGLE     = PARAMS['MAX_ANGLE']
TRACK_CLASS   = PARAMS['TRACK_CLASS']   # 兼容老引用
DEADZONE      = PARAMS['DEADZONE']
STEP_RATIO    = PARAMS['STEP_RATIO']
CMD_COOLDOWN  = PARAMS['CMD_COOLDOWN']
ANGLE_EPS     = PARAMS['ANGLE_EPS']
CONF_THRESH   = PARAMS['CONF_THRESH']
UART_PORT     = PARAMS['UART_PORT']
UART_BAUD     = PARAMS['UART_BAUD']

# 坐标系：center_x=320 对应角度=0°
FRAME_W      = 640
CENTER_X     = FRAME_W // 2

# ──── BPU 模型 ────
def bgr2nv12_opencv(image):
    h, w = image.shape[:2]
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420)
    y = yuv[:h, :w]
    uv = yuv[h:, :w].reshape(h//2, w//2, 2).transpose(1, 0, 2).reshape(h//2, w)
    return np.ascontiguousarray(np.vstack([y, uv]).ravel())
def letterbox(img, new_shape=640, color=(114, 114, 114)):
    """保比例 resize + 灰边填充，匹配 YOLOv5 训练时的预处理。
    解决画框系统性偏移：直接 cv2.resize 拉伸会破坏长宽比。
    返回 (img_letterboxed, ratio, dx, dy)：
      - ratio  : 缩放比（保比例，r ≤ 1）
      - dx, dy : 左/上灰边偏移（int, 0 ≤ dx < new_shape）
    画框还原公式：orig_xy = (model_xy - 灰边) / ratio
    """
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    pad_w, pad_h = new_shape - new_w, new_shape - new_h
    left, top = pad_w // 2, pad_h // 2
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    img_padded = cv2.copyMakeBorder(img_resized, top, pad_h - top, left, pad_w - left,
                                     cv2.BORDER_CONSTANT, value=color)
    return img_padded, r, left, top


def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-20,20)))

print(f"⏳ 加载模型: {MODEL_PATH}")
models = dnn.load(MODEL_PATH)
model = models[0]

# 通用逻辑：自动判断 output[1]（box）的通道在 dim[1] (NCHW) 还是 dim[-1] (NHWC)
_s1 = list(model.outputs[1].properties.shape)
if _s1[1] == 4:   # NCHW: box 通道在 dim[1]
    box_idx = {80: 0, 40: 2, 20: 4}
    cls_idx = {80: 1, 40: 3, 20: 5}
    CLS_AXIS = 0  # 5 通道那一维（dim=0）
else:                # NHWC: box 通道在 dim[-1]
    box_idx = {80: 1, 40: 3, 20: 5}
    cls_idx = {80: 0, 40: 2, 20: 4}
    CLS_AXIS = -1  # 5 通道在最后一维

def decode_detections(outs):
    boxes, scores, cls_ids = [], [], []
    for stride, H in [(8,80),(16,40),(32,20)]:
        ct = outs[cls_idx[H]].buffer[0]
        bt = outs[box_idx[H]].buffer[0]
        cls_probs = sigmoid(ct)
        max_prob = cls_probs.max(axis=CLS_AXIS)  # 5 类别那一维取 max → (H, W)
        best_cls = cls_probs.argmax(axis=CLS_AXIS)  # 取类别 argmax → (H, W)
        gx, gy = np.meshgrid(np.arange(H), np.arange(H), indexing='ij')
        # 新逻辑（test_new5.py 验证 test1.jpg 准）：sigmoid(cx,cy) + grid*stride，w/h 直接 *stride
        gy, gx = np.meshgrid(np.arange(H), np.arange(H), indexing='ij')
        cx = (sigmoid(bt[..., 0]) + gx) * stride
        cy = (sigmoid(bt[..., 1]) + gy) * stride
        # === 根因修复: YOLOv8 box 公式 (之前是 YOLOv5 简化版 bt*stride, 框偏小且帧间抖) ===
        # YOLOv8: w = (2 * sigmoid(bt))**2 * stride, 同样 h
        w = (2 * sigmoid(bt[..., 2])) ** 2 * stride
        h = (2 * sigmoid(bt[..., 3])) ** 2 * stride
        mask = max_prob.ravel() > PARAMS['CONF_THRESH']
        if not mask.any(): continue
        x1 = (cx - w/2).ravel()[mask]
        y1 = (cy - h/2).ravel()[mask]
        x2 = (cx + w/2).ravel()[mask]
        y2 = (cy + h/2).ravel()[mask]
        sc = max_prob.ravel()[mask]
        cl = best_cls.ravel()[mask]
        boxes.append(np.stack([x1, y1, x2, y2], axis=1))
        scores.append(sc)
        cls_ids.append(cl)
    if not boxes: return []
    boxes = np.vstack(boxes)
    scores = np.concatenate(scores)
    cls_ids = np.concatenate(cls_ids)
    order = np.argsort(-scores)
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(tuple(boxes[i].tolist()) + (float(scores[i]), int(cls_ids[i])))
        if len(order) == 1: break
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_o = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        iou = inter / (area_i + area_o - inter + 1e-7)
        order = order[1:][iou <= 0.20]   # int8 量化误差大，NMS 阈值从 0.45 降到 0.2 合并同目标多个框
    return keep

# ──── 电机控制 ────
motor = None
current_angle = 0
last_cmd_time = 0
motor_lock = threading.Lock()
# USB 设备是否插入（watchdog 每 5s 扫一次，仅给前端展示，不自动重连）
usb_present = False
# 方向防抖状态机
_stable_dir = 0       # -1/0/+1：当前目标的运动方向
_stable_count = 0     # 连续同向帧计数，达到 STABLE_FRAMES 才允许下发
_last_seen_target_at = None
_idle_home_sent = False
# mega2560 启动/回零时间戳（reader 线程收到帧时设）
mega_boot_at = None   # 收到"上电开始自动回零"的时间
mega_homed_at = None  # 收到"上电回零完成"的时间
mega_last_rx_at = None

# === 1D Kalman 滤波 state = [x, v]（位置+速度）===
# 失检期（best_det=None）只 predict 不 update，用 v 项惯性继续走
# 重新检测时 update 修正；速度项让目标快速移动 / 失检-重现段跟踪更稳
_kf_x = None      # 2D 状态向量 [位置, 速度]
_kf_p = None      # 2x2 协方差矩阵
_kf_y = None      # 2D 状态 [y, v_y] (Y 方向独立 KF, 共用 Q 噪声)
_kf_p_y = None

# === 画框 w/h 帧间 EMA 平滑 ===
_ema_w = 0.0
_ema_h = 0.0
_EMA_SIZE_ALPHA = 0.4

# === Kalman 常量（不依赖 PARAMS，模块级一次性算好）===
KF_DT = 1.0 / 30.0
KF_F  = np.array([[1.0, KF_DT], [0.0, 1.0]])   # 状态转移: x' = x + v·dt
KF_H  = np.array([[1.0, 0.0]])                  # 观测: 只测位置

def _usb_watchdog():
    """每 5s 扫一次 USB 节点。**不自动重连**，只更新 usb_present 给前端展示。
    想重连时通过 /api/set 改 UART_PORT 或 /api/manual 触发 open_motor。"""
    global usb_present
    while True:
        try:
            usb_present = os.path.exists(PARAMS['UART_PORT'])
        except Exception:
            usb_present = False
        time.sleep(5)

def open_motor(port=None, baud=None):
    """打开/重开电机串口。port/baud 不传则用 PARAMS 当前值。
    返回 'ok' / 'fail'。**启动前先检查 USB 节点是否存在**，未插入直接静默 return 'fail'，
    不会抛异常轰炸日志；纯视觉模式照常运行。"""
    global motor
    port = port or PARAMS['UART_PORT']
    baud = baud or PARAMS['UART_BAUD']
    PARAMS['UART_PORT'] = port
    PARAMS['UART_BAUD'] = baud
    # 关闭旧句柄
    if motor is not None:
        try: motor.close()
        except: pass
    # === 新增：先检查 USB 节点是否存在 ===
    if not os.path.exists(port):
        # 用户配置的端口不存在 → 按 CH340 VID:PID 自动扫描
        detected = find_mega_port()
        if detected and os.path.exists(detected):
            print(f"🔎 配置的 {port} 不在，但按 VID:PID 找到 CH340 → {detected}，自动切换。", flush=True)
            port = detected
            PARAMS['UART_PORT'] = port
        else:
            try:
                siblings = sorted(p for p in os.listdir('/dev') if p.startswith('ttyUSB') or p.startswith('ttyACM'))
                hint = f"当前 /dev 下发现: {siblings}" if siblings else "当前 /dev 下无 ttyUSB*/ttyACM* 节点"
            except Exception:
                hint = "无法列出 /dev"
            motor = None
            print(f"❌ USB 设备未插入 ({port})，{hint}。纯视觉模式运行。", flush=True)
            return 'fail'
    try:
        motor = serial.Serial(port, baud, timeout=0.3)
        print(f"🔌 电机串口已打开: {port} @ {baud}")
        return 'ok'
    except Exception as e:
        motor = None
        print(f"⚠️ 串口打开失败 ({port}): {e}（节点存在但打开失败：权限/被占用？）。纯视觉模式。", flush=True)
        return 'fail'

def configure_orchestrator_hooks(motor_handler=None, event_handler=None, audio_bins_provider=None):
    global _external_motor_handler, _external_event_handler, _external_audio_bins_provider
    _external_motor_handler = motor_handler
    _external_event_handler = event_handler
    _external_audio_bins_provider = audio_bins_provider


def push_camera_frame(frame):
    global shared_frame, frame_seq
    with shared_lock:
        shared_frame = frame
        frame_seq += 1
    frame_ready.set()


def send_motor_cmd(angle):
    if _external_motor_handler is not None:
        try:
            _external_motor_handler(float(angle))
        except Exception as e:
            print(f"External motor callback failed: {e}", flush=True)
        return
    """向 mega2560 下发 SimpleFOC Commander `T<angle>` 指令。
    协议格式：`T<角度整数>\\n`，例：T30\\n / T-45\\n
    串口未打开时静默跳过。"""
    if not should_i_drive_servo():
        return  # 6/19 dual-mode: TRACK_MODE='audio' 时 visual 不写舵机
    if motor is None:
        return
    try:
        motor.write(f"T{int(round(angle))}\n".encode())
    except Exception as e:
        print(f"⚠️ 电机写入失败: {e}")

def force_motor_angle(target_deg):
    """Direct motor command for manual/debug control, bypassing tracking filters."""
    global current_angle, last_cmd_time
    target = max(PARAMS['MIN_ANGLE'], min(PARAMS['MAX_ANGLE'], float(target_deg)))
    with motor_lock:
        current_angle = target
        send_motor_cmd(target)
        last_cmd_time = time.time()
    return target


def reset_mega_runtime_state(clear_log=False):
    global mega_boot_at, mega_homed_at, mega_last_rx_at
    mega_boot_at = None
    mega_homed_at = None
    mega_last_rx_at = None
    if clear_log:
        mega_log.clear()


def _decode_mega_line(raw_line):
    """Decode Mega serial output from mixed encodings used by legacy sketches."""
    for encoding in ("utf-8", "gb18030", "latin-1"):
        try:
            return raw_line.decode(encoding).strip()
        except UnicodeDecodeError:
            continue
    text = raw_line.decode("utf-8", "ignore").strip()
    return text if text else raw_line.hex(" ")


def _serial_reader_loop():
    """后台读 mega2560 回包。**新 mega 协议**：
      当前相对角度(°): 20.0  转速: 0.1000        ← SimpleFOC 调试行（每 500ms）
      上电开始自动回零至 0° ...                 ← 启动横幅 → mega_boot_at
      上电回零完成，当前位置：0°                ← 回零完成 → mega_homed_at
      指令：T0 回零 | T90 正向最大 | T-90 反向最大  ← initFOC 完毕
    """
    global current_angle, mega_boot_at, mega_homed_at, mega_last_rx_at
    import re
    # === 新协议正则 ===
    # 兼容旧中文日志和新英文结构化日志。
    angle_pat = re.compile(r'(?:当前相对角度\(°\):|REL_ANGLE_DEG:|REL_DEG=)\s*(-?\d+(?:\.\d+)?)')
    legacy_angle_pat = re.compile(r'[:：]\s*(-?\d+(?:\.\d+)?)\s+(?:转速|VEL:)')
    boot_pat  = re.compile(r'(?:上电开始自动回零|Startup zero lock|MOT:BOOT_HOME_START|MOT:PRE_INITFOC|MOT:POST_INITFOC)')
    homed_pat = re.compile(r'(?:上电回零完成|Startup zero lock done|ZERO_LOCK_TIMEOUT|MOT:HOMED|MOT:HOME_TIMEOUT_USE_CURRENT_ZERO|MOT:HOME_TIMEOUT|MOT:READY|Motor ready\.)')
    while True:
        if motor is None:
            time.sleep(0.2); continue
        try:
            line = motor.readline()
            if not line:
                continue
            txt = _decode_mega_line(line)
            if not txt:
                continue
            mega_last_rx_at = time.time()
            # 角度上报（每 500ms 一行）
            m = angle_pat.search(txt) or legacy_angle_pat.search(txt)
            if m:
                mega_log.append(txt)   # 调试行也入 ring buffer
                try:
                    val = float(m.group(1))
                    with motor_lock:
                        current_angle = val
                except ValueError:
                    pass
                # 同时启动 / 回零信号也可能在同一行后面
                if boot_pat.search(txt) and mega_boot_at is None:
                    mega_boot_at = time.time()
                    print(f"✅ mega2560 启动: {txt}", flush=True)
                if homed_pat.search(txt) and mega_homed_at is None:
                    mega_homed_at = time.time()
                    print(f"✅ mega2560 回零完成: {txt}", flush=True)
                continue
            # 启动 / 回零信号（独立行）
            if boot_pat.search(txt) and mega_boot_at is None:
                mega_boot_at = time.time()
                print(f"✅ mega2560 启动: {txt}", flush=True)
            elif homed_pat.search(txt) and mega_homed_at is None:
                mega_homed_at = time.time()
                print(f"✅ mega2560 回零完成: {txt}", flush=True)
            else:
                # 非协议帧（SimpleFOC 启动横幅 / commander 响应 / 上电提示）原样 print
                if txt:
                    print(f"[mega] {txt}", flush=True)
                    mega_log.append(txt)   # 全部非协议帧也入 ring buffer
        except Exception:
            time.sleep(0.1)

def set_motor_angle(target_deg):
    """朝目标角度逼近：限幅 + 步进 + 冷却 + 方向防抖。
    方向防抖：连续 STABLE_FRAMES 帧同向才下发，避免 BPU 检测框边缘抖动
    导致电机持续微步 → 过热。"""
    global current_angle, last_cmd_time, _stable_dir, _stable_count
    if not PARAMS['TRACK_ENABLED']:
        return
    if mega_boot_at is not None and mega_homed_at is None:
        return
    target = max(PARAMS['MIN_ANGLE'], min(PARAMS['MAX_ANGLE'], target_deg))

    with motor_lock:
        now = time.time()
        if now - last_cmd_time < PARAMS['CMD_COOLDOWN']:
            return
        delta = target - current_angle
        if abs(delta) < PARAMS['ANGLE_EPS']:
            return
        # === 方向防抖：连续 STABLE_FRAMES 帧同向才下发 ===
        new_dir = 1 if delta > 0 else -1
        if new_dir == _stable_dir:
            _stable_count += 1
        else:
            _stable_dir = new_dir
            _stable_count = 1
        if _stable_count < PARAMS['STABLE_FRAMES']:
            return
        # 通过所有防抖 → 下发

        step = delta * PARAMS['STEP_RATIO']
        current_angle += step
        current_angle = max(PARAMS['MIN_ANGLE'], min(PARAMS['MAX_ANGLE'], current_angle))
        send_motor_cmd(current_angle)
        last_cmd_time = now

# ──── Flask 推流 ────
app = Flask(__name__)
frame_lock = threading.Lock()
current_frame = None
current_fid = 0

def cap_thread():
    """v6 风格: 独立 grab 摄像头，把最新帧写入 shared_frame。
    主循环不再被 V4L2 cap.read 阻塞。"""
    global shared_frame, frame_seq
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        print("❌ 摄像头打不开")
        return
    print(f"📷 摄像头已开启 (cap_thread)")
    for _ in range(5):
        cap.read()  # 暖机
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue
        with shared_lock:
            shared_frame = frame
            frame_seq += 1
        frame_ready.set()

def main_loop():
    """v6 风格: 拿 shared_frame -> BPU 推理 + 简单画框 (无 X 镜像/无追踪选择/无追踪线) + HUD + imencode -> current_frame
    舵机不参与追踪 (只通过 /api/motor、/api/manual、/api/lock 手动控制)。"""
    global current_frame, current_fid, fps_display, _kf_x, _kf_p, _kf_y, _kf_p_y, _ema_w, _ema_h, _last_inf_ms, _last_obj_count, _last_target_name, _last_seen_target_at, _idle_home_sent
    last_seq = -1
    _fps_t = time.time()
    _fc = 0
    fps_display = 0.0
    _last_inf_ms = 0.0
    _last_obj_count = 0
    _last_target_name = '-'
    while True:
        frame_ready.wait()
        frame_ready.clear()
        with shared_lock:
            cur_frame = shared_frame
            cur_seq = frame_seq
        if cur_frame is None or cur_seq == last_seq:
            time.sleep(0.001)
            continue
        last_seq = cur_seq
        frame = cur_frame.copy()

        # === BPU 推理 (沿用 visual_track 视觉处理函数) ===
        t0 = time.time()
        img640, r, dx, dy = letterbox(frame, 640)
        nv12 = bgr2nv12_opencv(img640)
        outs = model.forward(nv12)
        dets = decode_detections(outs)
        inf_ms = (time.time() - t0) * 1000

        # === v6 风格画框 + 简单选追踪 (无 X 镜像, 不画追踪线) ===
        n = 0
        best_det = None
        best_conf = 0
        detections_summary = []
        # === BOX_SCALE 经验系数: 1.0=raw(BPU 框偏小, 因 decode 公式), 1.15≈真值 ===
        # 调 PARAMS['BOX_SCALE'] 实时改; KF 估的是 raw 中心, 不被框大小影响
        box_scale = float(PARAMS.get('BOX_SCALE', 1.15))
        for d in dets:
            x1, y1, x2, y2, sc, cid = d
            ox1 = int((x1 - dx) / r)
            oy1 = int((y1 - dy) / r)
            ox2 = int((x2 - dx) / r)
            oy2 = int((y2 - dy) / r)
            ox1 = max(0, min(ox1, frame.shape[1] - 1))
            oy1 = max(0, min(oy1, frame.shape[0] - 1))
            ox2 = max(0, min(ox2, frame.shape[1] - 1))
            oy2 = max(0, min(oy2, frame.shape[0] - 1))
            cls_name = CLS_NAMES[int(cid)] if int(cid) < len(CLS_NAMES) else f'cls_{int(cid)}'
            detections_summary.append({
                'cls_name': cls_name,
                'score': float(sc),
                'box': [int(ox1), int(oy1), int(ox2), int(oy2)],
            })
            # 算 raw 中心 + w/h (后续给 KF + EMA 用, 画框本身用 KF 估的值 → 不跳)
            cx_b = (ox1 + ox2) // 2
            cy_b = (oy1 + oy2) // 2
            w_b  = ox2 - ox1
            h_b  = oy2 - oy1
            # === w/h EMA 平滑 (防止画框大小帧间跳) ===
            if _ema_w <= 0:  # 冷启动
                _ema_w, _ema_h = float(w_b), float(h_b)
            else:
                _ema_w = (1 - _EMA_SIZE_ALPHA) * _ema_w + _EMA_SIZE_ALPHA * float(w_b)
                _ema_h = (1 - _EMA_SIZE_ALPHA) * _ema_h + _EMA_SIZE_ALPHA * float(h_b)
            # === 画框中心用 KF 估的 (而非 raw BPU 中心, 核心修复) ===
            cx_d = float(_kf_x[0]) if _kf_x is not None else cx_b
            cy_d = float(_kf_y[0]) if _kf_y is not None else cy_b
            sx = max(0, min(int(cx_d - _ema_w * box_scale / 2), frame.shape[1] - 1))
            sy = max(0, min(int(cy_d - _ema_h * box_scale / 2), frame.shape[0] - 1))
            ex = max(0, min(int(cx_d + _ema_w * box_scale / 2), frame.shape[1] - 1))
            ey = max(0, min(int(cy_d + _ema_h * box_scale / 2), frame.shape[0] - 1))
            cv2.rectangle(frame, (sx, sy), (ex, ey), (0, 255, 0), 2)
            # label (v6 风格: 黑底绿字) - 用放大后的 (sx, sy)
            label = f'{cls_name} {sc:.2f}'
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ty = max(0, sy - 4)
            cv2.rectangle(frame, (sx, max(0, ty - th - 2)), (sx + tw + 4, ty + 2), (0, 255, 0), -1)
            cv2.putText(frame, label, (sx + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            # 选追踪目标 (用 raw 中心 — KF 估的精度不损失, 框大小不影响追踪)
            tc = PARAMS['TRACK_CLASS']
            if tc is None or cls_name == tc:
                if sc > best_conf:
                    best_conf = sc
                    best_det = (cx_b, cy_b, cls_name, sc)   # 加 cy_b, 给 KF_y 用
            n += 1

        # === 1D Kalman 滤波 (state=[x, v]) — 失检期只 predict 不 update ===
        if _kf_x is None:
            if best_det is not None:
                _kf_x = np.array([float(best_det[0]), 0.0])   # 冷启动: v=0
                _kf_p = np.eye(2) * 100.0                      # 高不确定
        else:
            # predict (每帧都做, 包括失检帧 — 靠 v 项惯性继续走)
            Q = np.diag([float(PARAMS['KF_Q_POS']), float(PARAMS['KF_Q_VEL'])])
            _kf_x = KF_F @ _kf_x
            _kf_x[1] *= float(PARAMS.get('KF_V_DECAY', 0.95))  # v 泄漏 → 防止过冲
            _kf_p = KF_F @ _kf_p @ KF_F.T + Q
            # update (只在有检测时 — 用 BPU 测量修正预测)
            if best_det is not None:
                z  = float(best_det[0])
                y  = z - (KF_H @ _kf_x)[0]                    # 创新 (innovation)
                S  = (KF_H @ _kf_p @ KF_H.T + np.array([[float(PARAMS['KF_R_MEAS'])]]))[0, 0]
                K  = (_kf_p @ KF_H.T) / S                     # 卡尔曼增益
                _kf_x = _kf_x + K.flatten() * y
                _kf_p = (np.eye(2) - K @ KF_H) @ _kf_p

        # === 1D Kalman Y 方向 (共用 Q 噪声; 失检期只 predict, 跟 KF_x 一样) ===
        # 画框 cy 用 _kf_y[0] → 解决 cy 帧间跳 (与 _kf_x 配套, 框完全 KF 平滑)
        if _kf_y is None:
            if best_det is not None:
                _kf_y = np.array([float(best_det[1]), 0.0])   # best_det[1] = cy_b
                _kf_p_y = np.eye(2) * 100.0
        else:
            Q_y = np.diag([float(PARAMS['KF_Q_POS']), float(PARAMS['KF_Q_VEL'])])
            _kf_y = KF_F @ _kf_y
            _kf_y[1] *= float(PARAMS.get('KF_V_DECAY', 0.95))  # y 方向同样衰减
            _kf_p_y = KF_F @ _kf_p_y @ KF_F.T + Q_y
            if best_det is not None:
                z_y  = float(best_det[1])
                y_y  = z_y - (KF_H @ _kf_y)[0]
                S_y  = (KF_H @ _kf_p_y @ KF_H.T + np.array([[float(PARAMS['KF_R_MEAS'])]]))[0, 0]
                K_y  = (_kf_p_y @ KF_H.T) / S_y
                _kf_y = _kf_y + K_y.flatten() * y_y
                _kf_p_y = (np.eye(2) - K_y @ KF_H) @ _kf_p_y

        # === 电机跟踪: 用 KF 位置 (而非 raw tx) 算 offset → set_motor_angle ===
        if _kf_x is not None:
            tx_filtered = float(_kf_x[0])
            offset = tx_filtered - CENTER_X
            if abs(offset) > PARAMS['DEADZONE']:
                target_angle = offset / CENTER_X * PARAMS['MAX_ANGLE']
                if PARAMS['TRACK_MODE'] == 'video':
                    set_motor_angle(target_angle)

        now = time.time()
        if best_det is not None:
            _last_seen_target_at = now
            _idle_home_sent = False
        elif (
            PARAMS.get('TRACK_ENABLED', True)
            and str(PARAMS.get('IDLE_HOME_ENABLED', True)).lower() not in ('0', 'false', 'off', 'no')
            and PARAMS.get('TRACK_MODE') == 'video'
            and mega_homed_at is not None
            and _last_seen_target_at is not None
            and not _idle_home_sent
            and now - _last_seen_target_at >= float(PARAMS.get('IDLE_HOME_TIMEOUT', 5.0))
        ):
            force_motor_angle(0.0)
            _idle_home_sent = True
            print(f"[track] idle timeout -> return motor to 0 deg after {now - _last_seen_target_at:.1f}s", flush=True)

        # === KF 追踪线 (cyan 竖线 + 中心圆点) — 直观对比 BPU 原始框 vs KF 跟哪 ===
        if _kf_x is not None:
            kf_px = int(_kf_x[0])
            cv2.line(frame, (kf_px, 0), (kf_px, frame.shape[0]), (255, 255, 0), 1, cv2.LINE_AA)
            cv2.circle(frame, (kf_px, frame.shape[0] // 2), 6, (255, 255, 0), 2)
            cv2.circle(frame, (kf_px, frame.shape[0] // 2), 2, (0, 0, 0), -1)

        # === 帧统计（用于 /api/status 展示，不画在视频上）===
        _last_inf_ms = inf_ms
        _last_obj_count = n
        if best_det is not None:
            _last_target_name = best_det[2]  # cls_name

        # === FPS 计数 (1s 滑窗) ===
        _fc += 1
        now = time.time()
        if now - _fps_t >= 1.0:
            fps_display = _fc / (now - _fps_t)
            _fps_t = now
            _fc = 0
            if n > 0:
                print(f"📊 FPS={fps_display:.1f} Inf={inf_ms:.0f}ms obj={n}", flush=True)

        # === imencode -> current_frame (供 gen() 推流) ===
        if _external_event_handler is not None:
            try:
                _external_event_handler({
                    'type': 'vision_detection',
                    'frame_seq': cur_seq,
                    'detections': detections_summary,
                    'best_target': None if best_det is None else {
                        'center_x': int(best_det[0]),
                        'center_y': int(best_det[1]),
                        'cls_name': best_det[2],
                        'score': float(best_det[3]),
                    },
                    'inference_ms': float(inf_ms),
                    'object_count': int(n),
                })
            except Exception as e:
                print(f"Vision event callback failed: {e}", flush=True)

        with frame_lock:
            _, current_frame = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            current_fid += 1


def gen():
    last_fid = -1
    while True:
        t0 = time.time()
        with frame_lock:
            fid = current_fid
            frame = current_frame
        if frame is None or fid == last_fid:
            time.sleep(0.002)
            continue
        last_fid = fid
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame.tobytes() + b'\r\n'
        elapsed = time.time() - t0
        if elapsed < 1.0/28:
            time.sleep(1.0/28 - elapsed)

# === 麦克风频谱推流 (SSE) - reSpeaker XVF3800 真实音频 ===
AUDIO_RATE   = 16000
AUDIO_HOP    = 2048    # 1024 太小 USB 带宽争抢致 reSpeaker 驱动 hang
AUDIO_BINS   = 32
_audio_bins  = [0.0] * AUDIO_BINS
_audio_lock  = threading.Lock()

def audio_thread():
    """sounddevice 抓 reSpeaker PCM -> FFT -> 32 频段 magnitudes -> 写 _audio_bins.
    device 0 = reSpeaker XVF3800. 失败不 crash 主进程."""
    global _audio_bins
    try:
        with sd.InputStream(samplerate=AUDIO_RATE, channels=1, dtype='int16',
                            device=0, blocksize=AUDIO_HOP) as stream:
            print(f"🎤 mic spectrum started: reSpeaker device 0 @ {AUDIO_RATE}Hz "
                  f"hop={AUDIO_HOP}", flush=True)
            while True:
                data, _ = stream.read(AUDIO_HOP)
                pcm = data[:, 0].astype(np.float32) / 32768.0
                fft = np.fft.rfft(pcm * np.hanning(AUDIO_HOP))
                mag = np.abs(fft[:AUDIO_BINS]) / AUDIO_HOP
                mag = np.log10(mag + 1e-6)
                m_min, m_max = float(mag.min()), float(mag.max())
                if m_max > m_min:
                    mag = (mag - m_min) / (m_max - m_min)
                else:
                    mag = np.zeros_like(mag)
                with _audio_lock:
                    _audio_bins = mag.tolist()
    except Exception as e:
        print(f"❌ mic spectrum thread died: {e}", flush=True)

@app.route('/audio_stream')
def audio_stream():
    """SSE 单向推流：每 50ms 发一次 32 频段 bin."""
    import json as _json
    def gen():
        while True:
            payload = {"bins": [0.0] * AUDIO_BINS}
            if _external_audio_bins_provider is not None:
                try:
                    provided = _external_audio_bins_provider() or {}
                    payload["bins"] = list(provided.get("bins", payload["bins"]))
                    if "updated_at" in provided:
                        payload["updated_at"] = provided["updated_at"]
                except Exception as e:
                    payload["error"] = f"audio provider failed: {e}"
            else:
                with _audio_lock:
                    payload["bins"] = list(_audio_bins)
            yield f"data: {_json.dumps(payload)}\n\n"
            time.sleep(0.05)
    return Response(gen(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache',
                             'X-Accel-Buffering': 'no',
                             'Connection': 'keep-alive'})
@app.route('/alarm')
def alarm():
    """代理到 alarm_server (port 8001) 获取真实 4x4 GPIO 压力数据。"""
    import urllib.request, json as _json
    try:
        req = urllib.request.urlopen('http://localhost:8001/alarm', timeout=1)
        return _json.loads(req.read())
    except Exception as e:
        return {'grid': [[0]*4 for _ in range(4)], 'error': str(e)}

@app.route('/api/status')
def api_status():
    """返回运行状态（JSON）。"""
    return {
        'params': PARAMS,
        'current_angle': current_angle,
        'usb_present':   usb_present,        # 物理层：USB 节点是否存在
        'motor_connected': motor is not None,
        'camera_device': str(CAMERA_ID),
        'mega_boot':  mega_boot_at is not None,
        'mega_homed': mega_homed_at is not None,
        'mega_boot_at':  mega_boot_at,
        'mega_homed_at': mega_homed_at,
        'mega_last_rx_at': mega_last_rx_at,
        'fps': round(fps_display, 1) if 'fps_display' in globals() else 0.0,
        'inf_ms': round(_last_inf_ms, 1) if '_last_inf_ms' in globals() else 0.0,
        'obj_count': _last_obj_count if '_last_obj_count' in globals() else 0,
        'last_target': _last_target_name if '_last_target_name' in globals() else '-',
        'stable_count': _stable_count,      # 调试：当前方向防抖计数
        'mega_log': list(mega_log)[-40:],   # 最近 40 行 mega 输出（按时间正序）
        'mega_port': PARAMS['UART_PORT'],   # 当前实际使用的串口路径
    }

@app.route('/api/set', methods=['POST'])
@app.route('/api/params', methods=['POST'])
def api_set():
    """改参数 / 切串口 / 启停跟踪。body: {key: value}。"""
    data = request.get_json(silent=True) or {}
    log = []
    for k, v in data.items():
        if k not in PARAMS:
            log.append(f'unknown:{k}')
            continue
        if k in ('TRACK_CLASS',) and v in ('', 'None', 'none', 'null'):
            v = None
        if k in ('DEADZONE', 'MAX_ANGLE', 'MIN_ANGLE', 'ANGLE_EPS', 'UART_BAUD'):
            v = int(v)
        if k in ('STEP_RATIO', 'CMD_COOLDOWN', 'CONF_THRESH',
                 'KF_Q_POS', 'KF_Q_VEL', 'KF_R_MEAS', 'BOX_SCALE', 'KF_V_DECAY',
                 'IDLE_HOME_TIMEOUT'):
            v = float(v)
        if k in ('TRACK_ENABLED', 'IDLE_HOME_ENABLED'):
            v = bool(v) if isinstance(v, bool) else str(v).lower() in ('1','true','on','yes')
        PARAMS[k] = v
        log.append(f'{k}={v}')
    # 切串口：port 或 baud 改了才重开
    if 'UART_PORT' in data or 'UART_BAUD' in data:
        result = open_motor(PARAMS['UART_PORT'], PARAMS['UART_BAUD'])
        log.append(f'open_motor={result}')
    # 6/19 dual-mode: 切 TRACK_MODE 时同步写 /tmp/tracking_mode.json (integrated_main / visual_track 共享)
    if 'TRACK_MODE' in data:
        import json as _json_mode
        try:
            with open('/tmp/tracking_mode.json', 'w') as _f_mode:
                _json_mode.dump({'TRACK_MODE': data['TRACK_MODE']}, _f_mode)
            log.append(f'mode_file={data["TRACK_MODE"]}')
        except Exception as _e_mode:
            log.append(f'mode_file_error={_e_mode}')
    return {'ok': True, 'applied': log, 'params': PARAMS}

@app.route('/api/manual', methods=['POST'])
def api_manual():
    """手动下发一个角度（不管跟踪开关，调试用）。body: {angle: 30}。"""
    data = request.get_json(silent=True) or {}
    try:
        angle = float(data.get('angle', 0))
    except (TypeError, ValueError):
        return {'ok': False, 'err': 'bad angle'}, 400
    force_motor_angle(angle)
    return {'ok': True, 'sent': angle, 'current_angle': current_angle, 'motor_connected': motor is not None}
@app.route('/api/lock', methods=['POST'])
def api_lock():
    """锁停/解除锁停电机。
    body: {locked: true/false}   也接受 {lock: true/false}  可选 hold_angle(默认 0)。
    锁定=TRACK_ENABLED=False 跟踪停止 + 主动下发到 hold_angle 让电机锁在那里。
    解锁=TRACK_ENABLED=True 恢复跟踪，current_angle 重新被跟踪接管。"""
    global current_angle
    data = request.get_json(silent=True) or {}
    if 'locked' in data:
        locked = bool(data['locked'])
    elif 'lock' in data:
        locked = bool(data['lock'])
    else:
        return {'ok': False, 'err': 'need locked=true/false'}, 400
    try:
        hold = float(data.get('hold_angle', 0))
    except (TypeError, ValueError):
        hold = 0.0
    PARAMS['TRACK_ENABLED'] = not locked
    with motor_lock:
        current_angle = hold
        send_motor_cmd(hold)   # 锁定时下发一次，电机锁到 hold_angle
    return {
        'ok': True,
        'locked': locked,
        'track_enabled': PARAMS['TRACK_ENABLED'],
        'hold_angle': hold,
        'motor_connected': motor is not None,
    }

@app.route('/api/motor', methods=['POST'])
def api_motor():
    """手动下发一个角度（语义别名，等价 /api/manual）。body: {angle: 30}。"""
    data = request.get_json(silent=True) or {}
    try:
        angle = float(data.get('angle', 0))
    except (TypeError, ValueError):
        return {'ok': False, 'err': 'bad angle'}, 400
    force_motor_angle(angle)
    return {'ok': True, 'sent': angle, 'current_angle': current_angle,
            'motor_connected': motor is not None,
            'locked': not PARAMS['TRACK_ENABLED']}
CAPTURE_DIR = "/tmp/captures"
os.makedirs(CAPTURE_DIR, exist_ok=True)
_capture_idx = 0
def _next_capture_name():
    global _capture_idx
    _capture_idx += 1
    ts = time.strftime("%Y%m%d_%H%M%S")
    return f"capture_{ts}_{_capture_idx:03d}"
@app.route('/api/capture', methods=['POST'])
def api_capture():
    """拍照：接收前端 canvas 抓到的 base64 图像，跑 BPU 推理，保存原图+画框图+JSON。"""
    import base64, json
    data = request.get_json(silent=True) or {}
    img_b64 = data.get('image', '')
    pressure = data.get('pressure', None)
    if not img_b64.startswith('data:image'):
        return {'ok': False, 'err': 'image must be data:image/...;base64,...'}, 400
    if ',' in img_b64:
        img_b64 = img_b64.split(',', 1)[1]
    try:
        img_bytes = base64.b64decode(img_b64)
        img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        return {'ok': False, 'err': f'decode failed: {e}'}, 400
    if frame is None:
        return {'ok': False, 'err': 'cv2.imdecode returned None'}, 400
    t0 = time.time()
    img640, r, dx, dy = letterbox(frame, 640)
    nv12 = bgr2nv12_opencv(img640)
    outs = model.forward(nv12)
    dets = decode_detections(outs)
    inf_ms = (time.time() - t0) * 1000
    detections = []
    for d in dets:
        x1, y1, x2, y2, sc, cid = d
        ox1, oy1 = int((x1 - dx) / r), int((y1 - dy) / r)
        ox2, oy2 = int((x2 - dx) / r), int((y2 - dy) / r)
        ox1 = max(0, min(ox1, frame.shape[1])); oy1 = max(0, min(oy1, frame.shape[0]))
        ox2 = max(0, min(ox2, frame.shape[1])); oy2 = max(0, min(oy2, frame.shape[0]))
        detections.append({'cls_id': int(cid), 'cls_name': CLS_NAMES[int(cid)],
                          'score': float(sc), 'box': [ox1, oy1, ox2, oy2]})
    frame_marked = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det['box']
        color = (0, 255, 0) if det['cls_name'] == 'hunter' else (0, 0, 255)
        cv2.rectangle(frame_marked, (x1, y1), (x2, y2), color, 2)
        label = f"{det['cls_name']}:{det['score']:.2f}"
        cv2.putText(frame_marked, label, (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    if pressure and len(pressure) == 4:
        gh, gw = 80, 80
        gx0 = frame_marked.shape[1] - gw - 10
        gy0 = frame_marked.shape[0] - gh - 10
        cell_w, cell_h = gw // 4, gh // 4
        for i in range(4):
            for j in range(4):
                cx, cy = gx0 + j * cell_w, gy0 + i * cell_h
                active = pressure[i][j] if i < len(pressure) and j < len(pressure[i]) else False
                color = (0, 0, 255) if active else (80, 80, 80)
                cv2.rectangle(frame_marked, (cx, cy), (cx + cell_w - 2, cy + cell_h - 2), color, -1)
        cv2.putText(frame_marked, 'pressure', (gx0, gy0 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    ts_str = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame_marked, ts_str, (10, frame_marked.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    name = _next_capture_name()
    f_base = f"{name}.jpg"
    f_det = f"{name}_detected.jpg"
    f_info = f"{name}.json"
    p_base = os.path.join(CAPTURE_DIR, f_base)
    p_det = os.path.join(CAPTURE_DIR, f_det)
    p_info = os.path.join(CAPTURE_DIR, f_info)
    cv2.imwrite(p_base, frame)
    cv2.imwrite(p_det, frame_marked)
    info = {'timestamp': ts_str, 'ts_unix': time.time(),
            'frame_size': [int(frame.shape[1]), int(frame.shape[0])],
            'files': {'base': f_base, 'detected': f_det, 'json': f_info},
            'detections': detections, 'det_count': len(detections),
            'inference_ms': round(inf_ms, 1), 'pressure_grid': pressure,
            'params': dict(PARAMS), 'current_angle': current_angle,
            'motor_connected': motor is not None,
            'mega_log_tail': list(mega_log)[-5:]}
    with open(p_info, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"📸 拍照保存: {f_base}  ({len(detections)} 个目标)", flush=True)
    return {'ok': True, 'files': info['files'],
            'det_count': len(detections), 'detections': detections,
            'inference_ms': round(inf_ms, 1), 'path': p_base}

@app.route('/control')
def control():
    return CONTROL_HTML
CONTROL_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>电机控制台 · 飞羽卫士</title>
<style>
  body{font-family:'Segoe UI',Arial,sans-serif;background:#0b1226;color:#eef5ff;margin:0;padding:24px;}
  h1{color:#00ff99;margin:0 0 8px 0;font-size:22px;}
  h2{color:#00ccff;margin:24px 0 12px 0;font-size:15px;border-left:3px solid #2effb2;padding-left:10px;}
  .row{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:8px 0;}
  .card{background:#16213e;border-radius:10px;padding:18px;max-width:780px;}
  button{background:#0e2b3b;border:1px solid #3ebd93;color:#e0f2fe;padding:9px 18px;border-radius:30px;cursor:pointer;font-size:14px;transition:.15s;}
  button:hover{background:#1e6b5e;box-shadow:0 0 6px #3effb0;}
  button.danger{border-color:#ff5577;}
  button.danger:hover{background:#6e1e2e;box-shadow:0 0 6px #ff6680;}
  input{background:#071126;border:1px solid #2effb2;color:#eef5ff;padding:9px 12px;border-radius:8px;font-size:14px;width:90px;text-align:center;}
  #state{background:#071126;border-radius:10px;padding:14px;font-family:monospace;font-size:13px;line-height:1.6;margin-top:14px;min-height:80px;}
  .led{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;}
  .on{background:#00ff99;box-shadow:0 0 6px #00ff99;}
  .off{background:#ff5577;box-shadow:0 0 6px #ff5577;}
</style>
</head>
<body>
  <h1>🎛 电机控制台</h1>
  <p style="color:#9aa6b8;font-size:13px;">锁定=跟踪关闭 + 电机主动下发到 hold_angle（默认 0°）。解锁=恢复跟踪。</p>

  <div class="card">
    <h2>🔒 锁停电机</h2>
    <div class="row">
      <button class="danger" onclick="lock(true)">🔒 锁停电机（hold 0°）</button>
      <button onclick="lock(false)">🔓 解除锁停，恢复跟踪</button>
      <span style="color:#9aa6b8;font-size:12px;">也可指定锁停位置：</span>
      <input id="lockAngle" type="number" value="0" step="1" min="-90" max="90"> °
      <button onclick="lockToAngle()">锁到指定角度</button>
    </div>

    <h2>🎯 手动控制电机</h2>
    <div class="row">
      <button onclick="move(-90)">⬅ -90°</button>
      <button onclick="move(-45)">⬅ -45°</button>
      <button onclick="move(-15)">⬅ -15°</button>
      <button onclick="move(0)">⏺ 回中 0°</button>
      <button onclick="move(15)">➡ 15°</button>
      <button onclick="move(45)">➡ 45°</button>
      <button onclick="move(90)">➡ 90°</button>
    </div>
    <div class="row">
      <span style="color:#9aa6b8;font-size:12px;">自定义角度：</span>
      <input id="customAngle" type="number" value="0" step="1" min="-90" max="90"> °
      <button onclick="moveCustom()">下发</button>
    </div>

    <h2>📊 实时状态</h2>
    <div id="state">点击任意按钮后会刷新…</div>

    <h2>📡 mega 启动 / 调试日志</h2>
    <div style="color:#9aa6b8;font-size:12px;margin:6px 0;">最近 40 行 [mega] 输出（每 1.5s 自动刷新）。mega 启动横幅、FOC 状态、SimpleFOC 调试行都在这里。</div>
    <pre id="megaLog" style="background:#020a18;color:#9eff8e;border-radius:8px;padding:10px 14px;font-size:12px;line-height:1.45;max-height:280px;overflow-y:auto;white-space:pre-wrap;word-break:break-all;margin:0;">等待 mega 输出…</pre>
  </div>

<script>
const $ = id => document.getElementById(id);
async function post(url, body){
  const r = await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body||{})});
  return r.json();
}
async function refresh(){
  try{
    const r = await fetch('/api/status'); const s = await r.json();
    const locked = !s.params.TRACK_ENABLED;
    const usb = s.usb_present ? '✅ 已插入' : '❌ 未插入';
    const motor = s.motor_connected ? '✅ 已连接' : '❌ 未连接';
    const boot = s.mega_boot ? '✅ 是' : '⏳ 否';
    const homed = s.mega_homed ? '✅ 是' : '⏳ 否';
    $('state').innerHTML =
      `<span class="led ${locked?'off':'on'}"></span>跟踪状态: <b>${locked?'🔒 锁停中':'🟢 跟踪中'}</b><br>`+
      `当前电机角度: <b>${s.current_angle.toFixed(1)}°</b><br>`+
      `USB mega: <b>${usb}</b> · 串口 <b>${s.mega_port}</b>: <b>${motor}</b><br>`+
      `mega 启动: <b>${boot}</b> · 回零完成: <b>${homed}</b> · 防抖计数: <b>${s.stable_count}</b>/${s.params.STABLE_FRAMES}<br>`+
      `TRACK_CLASS: <b>${s.params.TRACK_CLASS??'所有'}</b> · DEADZONE: <b>${s.params.DEADZONE}</b><br>`+
      `MAX_ANGLE: <b>${s.params.MAX_ANGLE}</b> · MIN_ANGLE: <b>${s.params.MIN_ANGLE}</b><br>`+
      `STEP_RATIO: <b>${s.params.STEP_RATIO}</b> · COOLDOWN: <b>${s.params.CMD_COOLDOWN}s</b>`;
    // mega 启动 / 调试日志面板
    const logEl = $('megaLog');
    const log = s.mega_log || [];
    if (log.length === 0){
      logEl.textContent = '(暂无输出)';
    } else {
      // 自动滚到底部
      const wasAtBottom = logEl.scrollTop + logEl.clientHeight >= logEl.scrollHeight - 30;
      logEl.textContent = log.join('\n');
      if (wasAtBottom) logEl.scrollTop = logEl.scrollHeight;
    }
  }catch(e){ $('state').innerText='状态拉取失败: '+e; }
}
async function lock(locked){ await post('/api/lock', {locked}); refresh(); }
async function lockToAngle(){
  const a = parseFloat($('lockAngle').value);
  await post('/api/lock', {locked:true, hold_angle: a});
  refresh();
}
async function move(a){ await post('/api/motor', {angle:a}); refresh(); }
async function moveCustom(){ const a = parseFloat($('customAngle').value); move(a); }
setInterval(refresh, 1500);
refresh();
</script>
</body>
</html>"""

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_only')
def video_only():
    """纯净视频页面：黑底 + 居中 MJPEG 流，不含任何面板/控制/状态。"""
    return VIDEO_ONLY_HTML

VIDEO_ONLY_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>实时视频 · 飞羽卫士</title>
<style>
  html,body{margin:0;padding:0;height:100%;width:100%;background:#000;overflow:hidden;}
  body{display:flex;justify-content:center;align-items:center;}
  img{max-width:100%;max-height:100%;object-fit:contain;display:block;}
  /* 右下角小水印，避免完全黑屏不知道在加载 */
  .badge{position:fixed;right:10px;bottom:8px;color:#3a3a3a;font-family:Arial,sans-serif;font-size:11px;letter-spacing:1px;pointer-events:none;}
</style>
</head>
<body>
  <img src="/video_feed" alt="live video">
  <div class="badge">/video_feed · MJPEG · 飞羽卫士</div>
</body>
</html>"""

@app.route('/')
def index():
    return load_dashboard_html()

def load_dashboard_html():
    """Serve the embedded dashboard by default; external HTML is optional."""
    if PREFER_EMBEDDED_DASHBOARD:
        return DASHBOARD_HTML
    if not os.path.exists(DASHBOARD_PATH):
        print(f"⚠️ 找不到新版 dashboard: {DASHBOARD_PATH}，fallback 到内置 dashboard")
        return DASHBOARD_HTML
    try:
        with open(DASHBOARD_PATH, 'r', encoding='utf-8') as f:
            html = f.read()
        if not html.strip():
            print(f"⚠️ 新版网页为空文件: {DASHBOARD_PATH}，fallback 到内置 dashboard")
            return DASHBOARD_HTML
        return html
    except Exception as e:
        print(f"⚠️ 新版网页读取失败 ({e})，fallback 到内置 dashboard")
        return DASHBOARD_HTML

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>飞羽卫士——濒危鸟类智能保护装置</title>
    <style>
        *{margin:0;padding:0;box-sizing:border-box;font-family:Arial}
        body{background:#0b1226;color:#fff}
        .container{max-width:1400px;margin:0 auto;padding:20px}
        .title{text-align:center;font-size:28px;color:#00ff99;margin:20px 0}
        .main{display:grid;grid-template-columns:6fr 4fr;gap:20px}
        .video-box{background:#16213e;border-radius:12px;padding:20px}
        .video-box h2{color:#00ccff;margin-bottom:15px}
        .video-place{width:100%;height:420px;background:#000;border-radius:8px;overflow:hidden}
        .video-place img{width:100%;height:100%;object-fit:contain;display:block}
        .panel{background:#16213e;border-radius:12px;padding:25px}
        .item{margin:14px 0;font-size:17px}
        .label{color:#a0b0c0}
        .value{color:#fff;font-weight:bold;margin-left:8px}
        .warn{font-size:24px;font-weight:bold;text-align:center;margin-top:20px}
    </style>
</head>
<body>
<div class="container">
    <h1 class="title">📸 飞羽卫士——濒危鸟类智能监测平台</h1>
    <div class="main">
        <div class="video-box">
            <h2>🔍 实时监控画面</h2>
            <div class="video-place">
                <img src="/video_feed" alt="实时画面">
            </div>
        </div>
        <div class="panel">
            <div class="item"><span class="label">AI识别：</span><span class="value" id="ai">RF-DETR 正常</span></div>
            <div class="item"><span class="label">音频状态：</span><span class="value" id="audio">环境声正常</span></div>
            <div class="item"><span class="label">监测目标：</span><span class="value" id="target">朱鹮</span></div>
            <div class="item"><span class="label">ToF测距：</span><span class="value" id="dis">0 m</span></div>
            <div class="item"><span class="label">云台状态：</span><span class="value" id="ptz">自动跟踪中</span></div>
            <div class="item"><span class="label">GPS定位：</span><span class="value" id="gps">108.90°E, 34.32°N</span></div>
            <div class="item"><span class="label">更新时间：</span><span class="value" id="time"></span></div>
            <div class="warn"><span id="warning" style="color:#00ff99">✅ 安全监控中</span></div>
        </div>
    </div>
</div>

<script>
function update(){
    let d = new Date()
    document.getElementById('time').innerText = d.toLocaleTimeString()
    
    let dis = (Math.random()*80+5).toFixed(1)
    document.getElementById('dis').innerText = dis + ' m'
    
    let warn = document.getElementById('warning')
    if(dis<20){
        warn.innerText='⚠ 威胁逼近！'
        warn.style.color='#ff4444'
    }else{
        warn.innerText='✅ 安全监控中'
        warn.style.color='#00ff99'
    }
}
setInterval(update,500)
update()
</script>
</body>
</html>"""

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>飞羽卫士 | 实时联动看板</title>
    <style>
        :root{
            --bg:#08111d;
            --panel:#102238;
            --line:rgba(119,181,255,.18);
            --text:#edf6ff;
            --muted:#93acc7;
            --accent:#49dcb1;
            --accent-2:#78c6ff;
            --danger:#ff5b6e;
            --shadow:0 18px 50px rgba(0,0,0,.32);
        }
        *{box-sizing:border-box}
        html,body{margin:0;min-height:100%;background:
            radial-gradient(circle at top left, rgba(73,220,177,.12), transparent 28%),
            radial-gradient(circle at top right, rgba(120,198,255,.16), transparent 30%),
            linear-gradient(180deg, #08111d 0%, #0b1523 48%, #08111d 100%);
            color:var(--text);font-family:"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif}
        body{padding:8px 10px 14px}
        .shell{max-width:1600px;margin:0 auto}
        .topbar{display:flex;align-items:center;justify-content:space-between;gap:16px;padding:4px 0 10px}
        .brand h1{margin:0;font-size:22px;font-weight:700;letter-spacing:.02em}
        .brand p{margin:4px 0 0;color:var(--muted);font-size:13px}
        .chips{display:flex;gap:8px;flex-wrap:wrap;justify-content:flex-end}
        .chip{padding:7px 12px;border-radius:999px;background:rgba(255,255,255,.04);border:1px solid var(--line);font-size:12px;color:var(--muted)}
        .chip strong{color:var(--text)}
        .layout{display:grid;grid-template-columns:minmax(0,1.45fr) minmax(340px,.75fr);gap:12px;align-items:start}
        .card{background:linear-gradient(180deg, rgba(20,43,69,.94), rgba(11,24,39,.98));border:1px solid var(--line);border-radius:20px;box-shadow:var(--shadow)}
        .media-card{padding:10px}
        .media-head{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:4px 6px 10px}
        .media-head h2{margin:0;font-size:16px}
        .media-head p{margin:4px 0 0;color:var(--muted);font-size:12px}
        .tab-switch{display:flex;gap:8px}
        .tab-btn{border:1px solid var(--line);background:rgba(255,255,255,.04);color:var(--muted);padding:10px 16px;border-radius:999px;cursor:pointer;font-size:13px;font-weight:600}
        .tab-btn.active{background:linear-gradient(135deg, rgba(73,220,177,.2), rgba(120,198,255,.2));color:var(--text);border-color:rgba(120,198,255,.42)}
        .stage{display:none}
        .stage.active{display:block}
        .video-wrap{position:relative;overflow:hidden;border-radius:18px;background:#000;min-height:64vh}
        .video-wrap img{display:block;width:100%;height:64vh;object-fit:cover;object-position:center top}
        .video-overlay{position:absolute;left:12px;right:12px;top:12px;display:flex;justify-content:space-between;gap:10px;pointer-events:none}
        .overlay-pill{padding:7px 11px;border-radius:999px;background:rgba(8,17,29,.72);backdrop-filter:blur(8px);font-size:12px;border:1px solid rgba(255,255,255,.08)}
        .video-foot{display:grid;grid-template-columns:1.15fr .85fr;gap:12px;margin-top:12px}
        .subcard{background:rgba(255,255,255,.03);border:1px solid var(--line);border-radius:18px;padding:14px}
        .subcard h3{margin:0 0 10px;font-size:14px}
        .subcard p.meta{margin:0 0 10px;color:var(--muted);font-size:12px}
        .pressure-map{position:relative;height:240px;border-radius:18px;overflow:hidden;background:
            radial-gradient(circle at 32% 28%, rgba(73,220,177,.22), transparent 28%),
            radial-gradient(circle at 72% 65%, rgba(255,184,77,.16), transparent 24%),
            linear-gradient(160deg, #18324f 0%, #102238 45%, #0c1826 100%)}
        .pressure-map::before{content:"";position:absolute;inset:14px;border-radius:26px;border:1px solid rgba(255,255,255,.08)}
        .pressure-grid{position:absolute;inset:18px;display:grid;grid-template-columns:repeat(4,1fr);grid-template-rows:repeat(4,1fr);gap:1px;background:rgba(255,255,255,.08);padding:1px;border-radius:22px;overflow:hidden}
        .pressure-cell{position:relative;background:rgba(10,24,39,.72);display:flex;align-items:flex-end;justify-content:flex-start;padding:8px;transition:.18s ease}
        .pressure-cell span{font-size:11px;color:rgba(237,246,255,.72);letter-spacing:.08em}
        .pressure-cell.active{background:linear-gradient(135deg, rgba(255,91,110,.95), rgba(255,143,103,.9));box-shadow:inset 0 0 0 1px rgba(255,255,255,.18)}
        .pressure-cell.active span{color:#fff}
        .pressure-legend{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px;color:var(--muted);font-size:12px}
        .legend-dot{width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:5px}
        .audio-grid{display:grid;grid-template-columns:1.15fr .85fr;gap:12px}
        .audio-hero{padding:18px;min-height:64vh;display:flex;flex-direction:column;justify-content:space-between}
        .audio-hero h3{margin:0;font-size:16px}
        .audio-status{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;margin-top:14px}
        .audio-metric{background:rgba(255,255,255,.03);border:1px solid var(--line);border-radius:16px;padding:12px}
        .audio-metric small{display:block;color:var(--muted);font-size:11px;margin-bottom:6px}
        .audio-metric strong{font-size:18px}
        .spectrum-wrap{margin-top:16px;padding:16px;border-radius:18px;background:linear-gradient(180deg, rgba(8,17,29,.72), rgba(13,27,42,.92));border:1px solid rgba(255,255,255,.06)}
        .spectrum{height:220px;display:flex;align-items:flex-end;gap:5px}
        .bar{flex:1;min-width:0;border-radius:999px 999px 4px 4px;background:linear-gradient(180deg, rgba(120,198,255,.95), rgba(73,220,177,.72));height:12px;transition:height .08s linear, opacity .12s ease;opacity:.6}
        .bar.hot{background:linear-gradient(180deg, rgba(255,184,77,.95), rgba(255,91,110,.82));opacity:1}
        .spectrum-note{margin-top:10px;color:var(--muted);font-size:12px}
        .prob-list{display:grid;gap:10px}
        .prob-row{display:grid;grid-template-columns:72px 1fr 50px;gap:10px;align-items:center;font-size:12px}
        .prob-bar{height:8px;background:rgba(255,255,255,.08);border-radius:999px;overflow:hidden}
        .prob-fill{height:100%;background:linear-gradient(90deg, rgba(73,220,177,.95), rgba(120,198,255,.95))}
        .side-card{padding:16px}
        .side-head{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:14px}
        .side-head h2{margin:0;font-size:16px}
        .status-list{display:grid;gap:10px}
        .status-item{display:flex;justify-content:space-between;gap:12px;padding:12px 14px;border-radius:16px;background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.05)}
        .status-item span{color:var(--muted);font-size:13px}
        .status-item strong{font-size:13px;text-align:right}
        .alert-box{margin-top:12px;padding:16px;border-radius:18px;background:linear-gradient(135deg, rgba(73,220,177,.14), rgba(120,198,255,.09));border:1px solid rgba(120,198,255,.14)}
        .alert-box.danger{background:linear-gradient(135deg, rgba(255,91,110,.22), rgba(255,184,77,.12));border-color:rgba(255,91,110,.24)}
        .alert-title{font-size:13px;color:var(--muted);margin-bottom:8px}
        .alert-value{font-size:24px;font-weight:700}
        .small{font-size:12px;color:var(--muted)}
        .footnote{padding-top:12px;font-size:12px;color:var(--muted)}
        @media (max-width: 1180px){
            .layout,.video-foot,.audio-grid{grid-template-columns:1fr}
            .video-wrap,.video-wrap img,.audio-hero{min-height:52vh;height:auto}
            .video-wrap img{height:52vh}
        }
        @media (max-width: 760px){
            body{padding:6px}
            .topbar{flex-direction:column;align-items:flex-start}
            .chips{justify-content:flex-start}
            .audio-status{grid-template-columns:1fr}
            .video-wrap,.video-wrap img{height:42vh;min-height:42vh}
            .pressure-map{height:220px}
        }
    </style>
</head>
<body>
<div class="shell">
    <div class="topbar">
        <div class="brand">
            <h1>飞羽卫士实时联动看板</h1>
            <p>视频、压力、音频与情感识别统一可视化</p>
        </div>
        <div class="chips">
            <div class="chip">模式 <strong id="modeChip">video</strong></div>
            <div class="chip">AI <strong id="aiChip">初始化中</strong></div>
            <div class="chip">舵机 <strong id="servoChip">待连接</strong></div>
            <div class="chip">更新时间 <strong id="timeChip">--:--:--</strong></div>
        </div>
    </div>
    <div class="layout">
        <div class="card media-card">
            <div class="media-head">
                <div>
                    <h2>多模态监控视图</h2>
                    <p>视频区已顶格显示；可切换到音频情感面板查看识别细节</p>
                </div>
                <div class="tab-switch">
                    <button class="tab-btn active" id="videoTabBtn" onclick="setTab('video')">视频视图</button>
                    <button class="tab-btn" id="audioTabBtn" onclick="setTab('audio')">音频视图</button>
                </div>
            </div>
            <div class="stage active" id="videoStage">
                <div class="video-wrap">
                    <img src="/video_feed" alt="实时视频流">
                    <div class="video-overlay">
                        <div class="overlay-pill">目标 <strong id="videoTarget">--</strong></div>
                        <div class="overlay-pill">推理 <strong id="inferMs">-- ms</strong></div>
                        <div class="overlay-pill">FPS <strong id="fpsText">--</strong></div>
                    </div>
                </div>
                <div class="video-foot">
                    <div class="subcard">
                        <h3>压力地图</h3>
                        <p class="meta">将原 4x4 网格整合成连续地图，报警区仍按原区域联动变红。</p>
                        <div class="pressure-map">
                            <div class="pressure-grid" id="pressureGrid"></div>
                        </div>
                        <div class="pressure-legend">
                            <div><span class="legend-dot" style="background:rgba(237,246,255,.35)"></span>正常区域</div>
                            <div><span class="legend-dot" style="background:linear-gradient(135deg,#ff5b6e,#ff8f67)"></span>触发报警</div>
                        </div>
                    </div>
                    <div class="subcard">
                        <h3>视觉联动概况</h3>
                        <div class="status-list">
                            <div class="status-item"><span>当前目标</span><strong id="targetText">--</strong></div>
                            <div class="status-item"><span>目标数量</span><strong id="objCountText">0</strong></div>
                            <div class="status-item"><span>跟踪模式</span><strong id="trackModeText">video</strong></div>
                            <div class="status-item"><span>电机角度</span><strong id="angleText">0°</strong></div>
                            <div class="status-item"><span>串口状态</span><strong id="megaText">待检测</strong></div>
                        </div>
                        <div class="footnote">视频模式下由视觉追踪主导舵机；切到音频模式时由声源定位主导。</div>
                    </div>
                </div>
            </div>
            <div class="stage" id="audioStage">
                <div class="audio-grid">
                    <div class="subcard audio-hero">
                        <div>
                            <h3>音频情感与声源定位</h3>
                            <p class="meta">来自音频识别线程的最近一次事件与实时频谱。</p>
                            <div class="audio-status">
                                <div class="audio-metric"><small>最近物种</small><strong id="audioSpecies">--</strong></div>
                                <div class="audio-metric"><small>情感状态</small><strong id="audioEmotion">--</strong></div>
                                <div class="audio-metric"><small>识别置信度</small><strong id="audioConfidence">--</strong></div>
                                <div class="audio-metric"><small>声源方向 / 目标角</small><strong id="audioAngle">--</strong></div>
                            </div>
                        </div>
                        <div class="spectrum-wrap">
                            <div class="spectrum" id="audioSpectrum"></div>
                            <div class="spectrum-note" id="audioNote">等待音频频谱流...</div>
                        </div>
                    </div>
                    <div class="subcard">
                        <h3>音频识别细节</h3>
                        <div class="status-list">
                            <div class="status-item"><span>最近事件时间</span><strong id="audioEventAge">--</strong></div>
                            <div class="status-item"><span>振幅</span><strong id="audioAmp">--</strong></div>
                            <div class="status-item"><span>情感相似度</span><strong id="audioEmotionSim">--</strong></div>
                            <div class="status-item"><span>分析循环次数</span><strong id="audioIterations">--</strong></div>
                        </div>
                        <div style="height:12px"></div>
                        <h3>类别概率</h3>
                        <div class="prob-list" id="probList"></div>
                        <div class="footnote">若未运行 orchestrator，音频情感字段会保持为空，但频谱仍可显示。</div>
                    </div>
                </div>
            </div>
        </div>
        <div class="card side-card">
            <div class="side-head">
                <h2>系统状态</h2>
                <div class="small" id="uptimeText">Uptime --</div>
            </div>
            <div class="status-list">
                <div class="status-item"><span>AI状态</span><strong id="aiText">初始化中</strong></div>
                <div class="status-item"><span>音频状态</span><strong id="audioStateText">等待事件</strong></div>
                <div class="status-item"><span>舵机连接</span><strong id="ptzText">待检测</strong></div>
                <div class="status-item"><span>摄像头帧数</span><strong id="cameraFramesText">--</strong></div>
                <div class="status-item"><span>事件队列</span><strong id="eventQueueText">--</strong></div>
                <div class="status-item"><span>控制队列</span><strong id="controlQueueText">--</strong></div>
            </div>
            <div class="alert-box" id="warningBox">
                <div class="alert-title">当前告警</div>
                <div class="alert-value" id="warningText">安全监控中</div>
                <div class="small" id="warningMeta">暂无压力或目标触发</div>
            </div>
            <div class="footnote">
                健康接口：<code>/orchestrator/health</code><br>
                线程接口：<code>/orchestrator/threads</code><br>
                音频频谱：<code>/audio_stream</code>
            </div>
        </div>
    </div>
</div>
<script>
const pressureGrid = document.getElementById('pressureGrid');
const spectrumEl = document.getElementById('audioSpectrum');
const probListEl = document.getElementById('probList');
const pressureCells = [];
const spectrumBars = [];
const probLabels = ['gun', 'Nipponia-nippon', 'snake', 'weasel', 'background'];
let lastAlarmGrid = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]];
for(let r=0;r<4;r++){
  for(let c=0;c<4;c++){
    const cell = document.createElement('div');
    cell.className = 'pressure-cell';
    cell.innerHTML = `<span>${String.fromCharCode(65+r)}${c+1}</span>`;
    pressureGrid.appendChild(cell);
    pressureCells.push(cell);
  }
}
for(let i=0;i<32;i++){
  const bar = document.createElement('div');
  bar.className = 'bar';
  spectrumEl.appendChild(bar);
  spectrumBars.push(bar);
}
function buildProbRows(){
  probListEl.innerHTML = '';
  probLabels.forEach((name, idx) => {
    const row = document.createElement('div');
    row.className = 'prob-row';
    row.innerHTML = `<span>${name}</span><div class="prob-bar"><div class="prob-fill" id="probFill${idx}" style="width:0%"></div></div><strong id="probText${idx}">0%</strong>`;
    probListEl.appendChild(row);
  });
}
buildProbRows();
function setText(id, text){
  const el = document.getElementById(id);
  if(el) el.textContent = text;
}
function setTab(tab){
  document.getElementById('videoStage').classList.toggle('active', tab === 'video');
  document.getElementById('audioStage').classList.toggle('active', tab === 'audio');
  document.getElementById('videoTabBtn').classList.toggle('active', tab === 'video');
  document.getElementById('audioTabBtn').classList.toggle('active', tab === 'audio');
}
function formatAge(ts){
  if(!ts) return '--';
  const diff = Math.max(0, Date.now()/1000 - ts);
  if(diff < 1) return '刚刚';
  if(diff < 60) return `${diff.toFixed(1)}s 前`;
  return `${(diff/60).toFixed(1)}min 前`;
}
async function fetchJson(url){
  const response = await fetch(url, {cache:'no-store'});
  if(!response.ok) throw new Error(`${url} -> ${response.status}`);
  return response.json();
}
function renderPressureMap(grid){
  lastAlarmGrid = grid;
  pressureCells.forEach((cell, idx) => {
    const row = Math.floor(idx / 4);
    const col = idx % 4;
    const active = !!(grid[row] && grid[row][col]);
    cell.classList.toggle('active', active);
  });
  const activeCount = grid.flat().filter(Boolean).length;
  if(activeCount > 0){
    document.getElementById('warningBox').classList.add('danger');
    setText('warningText', `压力报警 ${activeCount} 区`);
    setText('warningMeta', '对应区域已变红，请结合视频与音频联动判断');
  }else{
    document.getElementById('warningBox').classList.remove('danger');
    setText('warningText', '安全监控中');
    setText('warningMeta', '暂无压力或目标触发');
  }
}
function renderSpectrum(bins){
  const list = Array.isArray(bins) ? bins : [];
  spectrumBars.forEach((bar, idx) => {
    const value = Math.max(0, Math.min(1, Number(list[idx] || 0)));
    bar.style.height = `${Math.max(2, value * 188)}px`;
    bar.classList.toggle('hot', value > 0.78);
  });
  const peak = list.length ? Math.max(...list) : 0;
  setText('audioNote', peak > 0.22 ? '频谱活跃，正在接收环境音变化。' : '频谱较平稳，当前环境音较弱。');
}
function renderProbabilities(probs){
  const list = Array.isArray(probs) ? probs : [];
  probLabels.forEach((_, idx) => {
    const value = Math.max(0, Math.min(1, Number(list[idx] || 0)));
    const fill = document.getElementById(`probFill${idx}`);
    const text = document.getElementById(`probText${idx}`);
    if(fill) fill.style.width = `${(value * 100).toFixed(1)}%`;
    if(text) text.textContent = `${(value * 100).toFixed(1)}%`;
  });
}
function applyStatus(status, health){
  const params = (status && status.params) || {};
  const audioStatus = health && health.audio ? health.audio : {};
  const audioDetection = audioStatus.last_detection || {};
  const cameraStatus = health && health.camera ? health.camera : {};
  const queues = health && health.queues ? health.queues : {};
  const system = health && health.system ? health.system : {};
  const mode = system.tracking_mode || params.TRACK_MODE || 'video';
  const target = status && status.last_target ? status.last_target : '--';
  const objCount = status && typeof status.obj_count !== 'undefined' ? status.obj_count : 0;
  const infMs = status && typeof status.inf_ms !== 'undefined' ? Number(status.inf_ms).toFixed(1) : '--';
  const fps = status && typeof status.fps !== 'undefined' ? Number(status.fps).toFixed(1) : '--';
  const servoConnected = status && status.motor_connected ? '已连接' : '未连接';
  const megaState = status && status.mega_homed ? '已回零' : (status && status.mega_boot ? '启动中' : '待检测');
  setText('modeChip', mode);
  setText('aiChip', objCount > 0 ? '检测中' : '在线');
  setText('servoChip', servoConnected);
  setText('timeChip', new Date().toLocaleTimeString());
  setText('videoTarget', target);
  setText('inferMs', `${infMs} ms`);
  setText('fpsText', fps);
  setText('targetText', target);
  setText('objCountText', String(objCount));
  setText('trackModeText', mode);
  setText('angleText', `${Number((status && status.current_angle) || 0).toFixed(1)}°`);
  setText('megaText', `${servoConnected} / ${megaState}`);
  setText('aiText', objCount > 0 ? `检测到 ${objCount} 个目标` : '在线待命');
  setText('audioStateText', audioDetection.species ? `${audioDetection.species} / ${formatAge(audioDetection.timestamp)}` : '等待音频事件');
  setText('ptzText', servoConnected);
  setText('cameraFramesText', cameraStatus.frame_count != null ? String(cameraStatus.frame_count) : '--');
  setText('eventQueueText', queues.event_queue != null ? String(queues.event_queue) : '--');
  setText('controlQueueText', queues.control_queue != null ? String(queues.control_queue) : '--');
  setText('uptimeText', system.uptime_s != null ? `Uptime ${system.uptime_s}s` : 'Uptime --');
  setText('audioSpecies', audioDetection.species || '--');
  setText('audioEmotion', audioDetection.emotion || '无明显情感');
  setText('audioConfidence', audioDetection.confidence != null ? `${(audioDetection.confidence * 100).toFixed(1)}%` : '--');
  if(audioDetection.doa_angle != null || audioDetection.target_angle != null){
    const doa = audioDetection.doa_angle != null ? `${Number(audioDetection.doa_angle).toFixed(0)}°` : '--';
    const tgt = audioDetection.target_angle != null ? `${Number(audioDetection.target_angle).toFixed(0)}°` : '--';
    setText('audioAngle', `${doa} / ${tgt}`);
  }else{
    setText('audioAngle', '--');
  }
  setText('audioEventAge', formatAge(audioDetection.timestamp));
  setText('audioAmp', audioDetection.amplitude != null ? Number(audioDetection.amplitude).toFixed(3) : '--');
  setText('audioEmotionSim', audioDetection.emotion_similarity != null ? Number(audioDetection.emotion_similarity).toFixed(3) : '--');
  setText('audioIterations', audioStatus.analysis_iterations != null ? String(audioStatus.analysis_iterations) : '--');
  renderProbabilities(audioDetection.probabilities || []);
  const hasPressure = lastAlarmGrid.flat().some(Boolean);
  if(hasPressure){
    document.getElementById('warningBox').classList.add('danger');
  }else if(target && target !== '-' && target !== '--'){
    document.getElementById('warningBox').classList.add('danger');
    setText('warningText', `视觉锁定 ${target}`);
    setText('warningMeta', `推理 ${infMs} ms | 目标 ${objCount} 个`);
  }else{
    document.getElementById('warningBox').classList.remove('danger');
  }
}
async function refreshDashboard(){
  const jobs = await Promise.allSettled([fetchJson('/api/status'), fetchJson('/alarm'), fetchJson('/orchestrator/health')]);
  const status = jobs[0].status === 'fulfilled' ? jobs[0].value : null;
  const alarm = jobs[1].status === 'fulfilled' ? jobs[1].value : null;
  const health = jobs[2].status === 'fulfilled' ? jobs[2].value : null;
  renderPressureMap(alarm && Array.isArray(alarm.grid) ? alarm.grid : [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]);
  applyStatus(status, health);
}
function initAudioStream(){
  try{
    const source = new EventSource('/audio_stream');
    source.onmessage = (event) => {
      try{
        const data = JSON.parse(event.data || '{}');
        renderSpectrum(data.bins || []);
      }catch(_err){}
    };
    source.onerror = () => setText('audioNote', '音频频谱流暂时断开，等待重连...');
  }catch(err){
    setText('audioNote', `音频频谱不可用: ${err}`);
  }
}
setTab('video');
initAudioStream();
refreshDashboard();
setInterval(refreshDashboard, 1500);
</script>
</body>
</html>"""

if __name__ == '__main__':
    # 自动拉起 alarm_server（GPIO 4x4 压力扫描，端口 8001）
    import subprocess as _sp
    _alarm_server_path = os.path.join(FILE_BASE_DIR, 'alarm_server.py')
    _sp.Popen([sys.executable, _alarm_server_path],
              stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
    time.sleep(0.5)
    open_motor()
    threading.Thread(target=_usb_watchdog,   daemon=True).start()  # USB 节点 5s 扫一次
    threading.Thread(target=_serial_reader_loop, daemon=True).start()
    threading.Thread(target=cap_thread, daemon=True).start()
    time.sleep(1.5)  # 等摄像头出第一帧
    threading.Thread(target=main_loop, daemon=True).start()
    threading.Thread(target=audio_thread, daemon=True).start()
    time.sleep(1)
    print(f"🚀 http://0.0.0.0:{STREAM_PORT}/video_feed")
    print(f"🎯 追踪模式: {'所有类别' if TRACK_CLASS is None else TRACK_CLASS}")
    app.run(host='0.0.0.0', port=STREAM_PORT, debug=False, threaded=True)
