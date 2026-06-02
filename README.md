# robot_car

`robot_car` 是面向 Orange Pi 5 Plus / RK3588 的智能小车工程。项目从旧目录 `/home/orangepi/robust_code` 重构而来，把原来分散在单文件脚本里的功能拆成了更清晰的工程结构：底盘运动、云台舵机、传感器、视觉追踪、语音交互、情绪识别和 Web 数据上传。

旧项目中的测试脚本、Notebook checkpoint、缓存文件和大模型文件不纳入 Git 仓库。模型文件保留在本机 `models/` 目录中，按需手动拷贝。

## 当前状态

已完成：

- TB6612 四电机底盘控制
- PCA9685 云台舵机控制
- 超声波测距和红外避障封装
- OpenCV 摄像头读取和 Haar 人脸追踪
- 面部情绪识别接口：`emotion-ferplus-8.onnx`
- 语音情绪识别接口：`SER.tflite`
- 本地 SenseVoice 语音转写、本地 RKLLM/Qwen 对话和本地 MeloTTS 语音播放
- Socket.IO 视频帧和情绪数据上传接口
- 主程序入口：`python -m robot_car.app.main`

运行完整语音功能前需要先启动本机 RKLLM Server，并确认本地 ASR/TTS 模型存在。

## 目录结构

```text
robot_car/
  robot_car/
    app/                 # 主程序和跟随控制策略
    audio/               # 录音、语音情绪识别、本地 Qwen/RKLLM 语音对话
    hardware/            # 电机、舵机、超声波、红外、LED/按键
    vision/              # 摄像头、人脸追踪、面部情绪识别
    web/                 # Socket.IO 数据和视频帧上传
    config.py            # 统一配置：引脚、模型路径、阈值、本地 LLM/Web 地址
    state.py             # 线程间共享状态
  assets/image/          # Haar XML，本机保留，不上传 Git
  data/                  # 运行时录音输出
  models/                # ONNX / RKNN / RKLLM / TFLite 模型，本机保留，不上传 Git
```

## 环境

当前 Orange Pi 上使用已有虚拟环境：

```bash
source /home/orangepi/pi/bin/activate
```

已验证核心环境：

```text
Python 3.9.18
periphery
smbus2
opencv-python / cv2
numpy
onnxruntime
pyaudio
soundfile
librosa
python-dotenv
pyserial
```

完整语音和 Web 功能还需要：

```bash
pip install sherpa_onnx requests python-socketio tflite-runtime
```

注意：当前板子上 `tensorflow` 导入会触发崩溃，所以语音情绪识别只使用 `tflite-runtime`。

## 配置

复制配置模板：

```bash
cd /home/orangepi/robot_car
cp .env.example .env
nano .env
```

常用环境变量：

```text
ROBOT_WEB_SERVER_URL=http://192.168.1.102:5000
ROBOT_CAMERA_BACKEND=opencv
ROBOT_CAMERA_DEVICE=0
ROBOT_SERVO_I2C_BUS=2
ROBOT_SERVO_I2C_ADDRESS=0x40
ROBOT_FOLLOW_ENABLED_ON_START=false
ROBOT_RKLLM_SERVER_URL=http://127.0.0.1:8080/rkllm_chat
ROBOT_LOCAL_ASR_ENABLED=true
ROBOT_LOCAL_TTS_ENABLED=true
```

如果 `i2cdetect` 看到 PCA9685 地址是 `0x60`，运行前可以覆盖：

```bash
export ROBOT_SERVO_I2C_ADDRESS=0x60
```

## 模型和资源

模型文件放在本机，不上传 GitHub。
`models/`、`assets/image/`、`data/` 和 `logs/` 这类运行目录会用 `.gitkeep` 保留目录结构；目录里的模型、XML、录音和日志文件仍然被 `.gitignore` 排除。

当前主程序直接使用：

```text
models/emotion-ferplus-8.onnx
models/SER.tflite
models/speech/sensevoice-small/model.int8.onnx
models/speech/sensevoice-small/tokens.txt
models/speech/vits-melo-tts-zh_en/model_int8.onnx
models/speech/vits-melo-tts-zh_en/lexicon.txt
models/speech/vits-melo-tts-zh_en/tokens.txt
models/llm/qwen3-vl-2b-instruct_w8a8_rk3588.rkllm
assets/image/haarcascade_frontalface_default.xml
assets/image/haarcascade_eye.xml
```

推荐模型目录：

```text
models/
  emotion-ferplus-8.onnx
  SER.tflite
  face/
    yolov8.onnx
    yolov8.rknn
    sface.onnx
    sface.rknn
    minixception.onnx
    minixception.rknn
  speech/
    sensevoice-small/
    vits-melo-tts-zh_en/
  llm/
    qwen3-vl-2b-instruct_w8a8_rk3588.rkllm
```

如果 `assets/image/` 里没有 Haar XML，程序会尝试使用 OpenCV 自带路径：

```text
/home/orangepi/pi/lib/python3.9/site-packages/cv2/data/
```

## 运行

完整本地语音对话需要先启动 RKLLM Server：

```bash
cd /home/orangepi/rknn-llm/examples/rkllm_server_demo/rkllm_server
python3 flask_server.py \
  --rkllm_model_path /home/orangepi/robot_car/models/llm/qwen3-vl-2b-instruct_w8a8_rk3588.rkllm \
  --target_platform rk3588
```

启动后接口地址默认为：

```text
http://127.0.0.1:8080/rkllm_chat
```

基础启动：

```bash
cd /home/orangepi/robot_car
source /home/orangepi/pi/bin/activate
python -m robot_car.app.main
```

没有配置实体按键时，直接启用跟随模式：

```bash
python -m robot_car.app.main --follow
```

先只验证视觉和硬件控制，不启用语音和 Web：

```bash
python -m robot_car.app.main --follow --no-voice --no-web
```

常用参数：

```bash
python -m robot_car.app.main --no-voice
python -m robot_car.app.main --no-web
python -m robot_car.app.main --skip-calibration
python -m robot_car.app.main --camera-backend opencv
```

## 硬件默认配置

硬件参数集中在 `robot_car/config.py`。

当前默认值：

```text
摄像头：/dev/video0，320x240
PCA9685：I2C bus 2，地址 0x40
云台舵机：pan=10，tilt=9
超声波：TRIG=100，ECHO=99
红外：left=101，right=35
跟随按键：默认未配置，建议先用 --follow
```

检查设备：

```bash
ls /dev/i2c-*
ls /dev/video*
i2cdetect -y 2
```

## Git 日常上传流程

不要把模型、API Key、运行日志、录音文件上传到 GitHub。`.gitignore` 已经忽略：

```text
.env
qwen.env
models/
assets/image/*.xml
data/*.wav
logs/
__pycache__/
```

确认远端地址：

```bash
git remote -v
```

本项目当前使用 GitHub SSH 443 端口，适合普通 22 端口不稳定时使用：

```text
ssh://git@ssh.github.com:443/Ming-yicheng/robot_car.git
```

每次修改代码或文档后，按下面流程上传：

```bash
# 1. 查看改了什么
git status --short

# 2. 检查是否误放了大文件或密钥
find . -path ./.git -prune -o -type f -size +50M -print
grep -R "sk-" . --exclude-dir=.git --exclude-dir=models

# 3. 添加要提交的文件
git add <文件1> <文件2>

# 如果确认 .gitignore 正确，也可以添加全部非忽略文件
git add -A

# 4. 提交
git commit -m "Describe your change"

# 5. 推送前先同步远端，避免覆盖别人提交
git pull --rebase

# 6. 上传到 GitHub
git push
```

如果不小心把模型、密钥或运行文件加入暂存区，先撤回暂存：

```bash
git restore --staged models/ assets/image/ .env qwen.env data/ logs/
```

推荐提交粒度：

```text
代码修改：git commit -m "Refine follow controller"
文档修改：git commit -m "Update README"
配置模板：git commit -m "Update environment example"
```
