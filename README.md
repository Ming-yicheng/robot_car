# robot_car

这是从 `/home/orangepi/robust_code` 重构出来的智能小车工程骨架。新工程只保留核心代码：底盘电机、PCA9685 舵机、超声波、红外、摄像头/人脸追踪、语音对话、情绪识别和 Web 数据上传。旧目录里的单文件测试脚本、Notebook checkpoint、缓存文件和大模型文件没有复制进来。

## 目录结构

```text
robot_car/
  robot_car/
    app/                 # 主程序和跟随控制策略
    audio/               # 录音、语音情绪、Qwen 语音对话
    hardware/            # 电机、舵机、传感器、机器人组合对象
    vision/              # 摄像头、人脸追踪、面部情绪
    web/                 # Socket.IO 数据和视频帧上传
    config.py            # 所有硬件引脚、模型路径、运行参数
    state.py             # 线程之间共享的小车状态
  assets/image/          # Haar 级联分类器，手动从旧目录或系统拷贝
  models/                # onnx/tflite/rknn/rkllm 等大模型，手动迁移
  data/                  # 运行时音频输出
```

## 运行

```bash
cd /home/orangepi/robot_car
source /home/orangepi/pi/bin/activate
python -m robot_car.app.main
```

常用参数：

```bash
python -m robot_car.app.main --follow       # 不接按键时，启动后直接进入跟随
python -m robot_car.app.main --no-voice     # 只跑视觉和轮子控制
python -m robot_car.app.main --no-web       # 不连接 Web 服务
python -m robot_car.app.main --skip-calibration
```

## 需要手动迁移的大文件

按你后续实际功能选择迁移，不需要一次全放进去：

```text
robust_code/face_recognition-master/yolov8.onnx
robust_code/face_recognition-master/minixception.onnx
robust_code/face_recognition-master/*.rknn
robust_code/Independent module/qwen3-vl-2b-instruct_w8a8_rk3588.rkllm
robust_code/Independent module/sensevoice-small/
robust_code/Independent module/vits-melo-tts-zh_en/
```

主函数里实际使用的模型路径已经统一到：

```text
robot_car/models/emotion-ferplus-8.onnx
robot_car/models/SER.tflite
robot_car/assets/image/haarcascade_frontalface_default.xml
robot_car/assets/image/haarcascade_eye.xml
```

如果 Haar XML 不放到 `assets/image/`，程序会优先尝试 OpenCV 自带路径。`emotion-ferplus-8.onnx` 和 `SER.tflite` 不存在时，对应情绪识别会自动禁用，主流程继续运行。

## 环境变量

不要把 API Key 写死在代码里。复制 `.env.example` 为 `.env` 或 `qwen.env`：

```bash
cp .env.example .env
nano .env
```

然后填入：

```text
DASHSCOPE_API_KEY=你的key
```

## 硬件配置

硬件引脚在 `robot_car/config.py` 中集中配置。当前默认值来自旧项目里相对成熟的完成版模块：

- TB6612 四电机：`finish module/tb6612_abcd_test.py`
- 超声波：`TRIG=100`，`ECHO=99`
- 红外：左 `101`，右 `35`
- 云台舵机：PCA9685，I2C bus `2`，地址默认 `0x40`

如果舵机没有动作，先在 Orange Pi 上检查：

```bash
ls /dev/i2c-*
i2cdetect -y 2
```

看到的地址如果是 `0x60`，可用环境变量覆盖：

```bash
export ROBOT_SERVO_I2C_ADDRESS=0x60
```
