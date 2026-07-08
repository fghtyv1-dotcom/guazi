# 飞羽卫士

飞羽卫士是一个面向野生动物巢区防护场景的边缘 AI 多元感知系统。项目围绕视频感知、音频感知、声源定位、压力触发、舵机控制和远程告警展开，用于对朱鹮巢区及周边风险目标进行实时监测与联动处置。

## 项目特点

- 视频目标检测与跟踪
- 音频事件识别
- 情感识别
- 麦克风阵列声源定位
- 压力传感矩阵触发
- 舵机云台控制
- 本地继电器告警
- ESP8266 远程蜂鸣器告警
- Web 实时看板

## 项目结构

```text
final/
├── README.md
├── orchestrator.py
├── visual_track.py
├── audio_adapter.py
├── integrated_main.py
├── video_infer.py
├── alarm_server.py
├── remote_alarm_bridge.py
├── relay_toggle.py
├── device_resolver.py
├── dashboard_final.html
├── feiyu (2).html
├── esp8266_remote_buzzer/
├── firmware_variants/
├── sound_data/
└── yamnet_compile/
```

## 核心脚本说明

### `orchestrator.py`

系统总控入口。

负责：

- 启动和管理视觉、音频、前端、告警等模块
- 统一调度摄像头、串口、继电器和远程告警桥接
- 汇总系统状态并执行联动逻辑

### `visual_track.py`

视觉感知与跟踪模块。

负责：

- 摄像头画面采集
- 目标检测与框选
- 舵机云台跟踪控制
- 视频推流和部分前端接口

### `audio_adapter.py`

音频线程适配层。

负责：

- 将音频识别模块接入总控框架
- 持续采集音频流
- 输出统一格式的音频事件
- 提供前端频谱数据

### `integrated_main.py`

音频识别与声源定位核心模块。

负责：

- 麦克风阵列音频采集
- YAMNet 特征提取
- 声音类别识别
- 情感识别
- 声源方向估计

### `video_infer.py`

离线视频推理脚本。

负责：

- 对本地视频或摄像头画面进行目标检测
- 输出带检测框的视频结果
- 支持独立联动告警测试

### `alarm_server.py`

压力传感矩阵服务。

负责：

- 扫描压力传感矩阵
- 维护压力报警网格状态
- 提供 `/alarm` 和 `/health` HTTP 接口

### `remote_alarm_bridge.py`

远程蜂鸣器桥接脚本。

负责：

- 轮询本地告警状态
- 将本地事件转发到 ESP8266 远程蜂鸣器
- 按事件类型设置不同报警策略

### `relay_toggle.py`

本地继电器控制模块。

负责：

- 控制继电器高低电平
- 按不同事件触发不同继电器动作

### `device_resolver.py`

设备自动发现模块。

负责：

- 自动寻找串口设备
- 自动寻找摄像头设备
- 自动寻找音频输入设备

## 前端文件

### `dashboard_final.html`

当前主要使用的实时看板页面，用于展示视频流、状态信息和告警信息。

### `feiyu (2).html`

早期版本前端页面。

## 数据与模型目录

### `yamnet_compile/`

存放音频识别相关模型和特征文件，例如：

- `species_classifier.pkl`
- `scaler.pkl`
- `emotion_database.npy`

### `sound_data/`

存放音频数据处理和情感识别相关脚本。

## 固件目录

### `esp8266_remote_buzzer/`

ESP8266 远程蜂鸣器固件。

### `firmware_variants/`

不同版本的控制板固件。

### `new/`

实验性控制板代码。

## 运行方式

### 启动整套系统

```bash
python3 orchestrator.py
```

### 只测试视觉模块

```bash
python3 visual_track.py
```

### 只测试离线视频推理

```bash
python3 video_infer.py --video your_video.mp4 --output result.mp4
```

### 只测试压力报警服务

```bash
python3 alarm_server.py
```

### 只测试远程蜂鸣器桥接

```bash
python3 remote_alarm_bridge.py
```

## 运行依赖

- Python 3
- OpenCV
- NumPy
- Flask
- pyserial
- sounddevice
- TensorFlow
- TensorFlow Hub
- scikit-learn
- gpiod
- Hobot.GPIO
- hobot_dnn

## 硬件组成

- 摄像头
- 麦克风阵列
- 舵机云台
- Mega 串口控制板
- 继电器模块
- 压力传感矩阵
- ESP8266 远程蜂鸣器
- 边缘 AI 计算板
