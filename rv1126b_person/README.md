# RV1126B 人体识别与异常行为检测

这个项目当前主要运行文件是：

```text
src/detect_track.py
```

它负责完成：

```text
摄像头/视频读取
YOLOv8 人体检测
人员 ID 跟踪
人数统计
ROI 区域入侵
异常停留
徘徊
多次靠近敏感区域
穿越警戒线
非工作时间人员出现
遮挡/贴近摄像头
快速奔跑
报警上报
```

## 本地推理、跟踪和告警

进入项目目录：

```bash
cd D:\.person_detect\rv1126b_person_edge_ai
```

运行摄像头实时检测：

```bash
python src/detect_track.py --show
```

如果要指定权重：

```bash
python src/detect_track.py --weights E:\dataset_person\runs\train\person_yolov8n_416_edge\weights\best.pt --show
```

如果要使用配置文件运行：

```bash
python src/detect_track.py --config configs/edge_runtime.yaml --show
```

程序默认会读取：

```text
configs/edge_runtime.yaml
```

所以平时推荐直接改这个配置文件。

## 当前一秒几帧

运行后，窗口左上角会显示 FPS，例如：

```text
FPS 14.9
```

意思是当前大约每秒处理 14.9 帧。

终端里也会每隔 2 秒打印一次：

```text
runtime_fps=14.90, process_every=1, imgsz=416
```

含义：

```text
runtime_fps=14.90   当前实际运行速度，大约每秒 14.90 帧
process_every=1     每 1 帧检测 1 帧，也就是不跳帧
imgsz=416           YOLO 输入尺寸是 416
```

如果运行卡顿，可以改：

```bash
python src/detect_track.py --imgsz 320 --process-every 2 --show
```

这样会更快，但检测精度会稍微下降。

## 运行时看哪里

窗口左上角：

```text
人数
禁区人数
FPS
抽帧间隔
```

窗口右上角：

```text
报警中心
```

报警会显示中文类别，例如：

```text
严重 · 区域入侵 ID 3
预警 · 异常停留 ID 5
严重 · 遮挡/贴近摄像头 ID 1
```

终端也会打印报警 JSON，例如：

```json
{"type":"intrusion","type_cn":"区域入侵","level":"critical","track_id":3,"time":12.44}
```

## ROI 区域在哪里改

改这个文件：

```text
configs/roi_example.json
```

当前格式：

```json
{
  "name": "restricted_area",
  "points": [
    [220, 160],
    [460, 160],
    [460, 360],
    [220, 360]
  ],
  "line_name": "fence_line",
  "line": [
    [220, 360],
    [460, 360]
  ]
}
```

参数含义：

```text
points    禁区 ROI 多边形坐标
line      警戒线两个端点坐标
name      禁区名称
line_name 警戒线名称
```

程序判断人的脚底中心点是否进入 ROI。

## 配置文件在哪里改

主要改这个文件：

```text
configs/edge_runtime.yaml
```

它里面的参数会被 `src/detect_track.py` 自动读取。

命令行参数优先级更高。例如配置文件里写了：

```yaml
running_speed: 200.0
```

但运行时写了：

```bash
python src/detect_track.py --running-speed 260 --show
```

那么实际使用的是 `260`。

## 常用运行命令

摄像头实时显示：

```bash
python src/detect_track.py --show
```

使用配置文件：

```bash
python src/detect_track.py --config configs/edge_runtime.yaml --show
```

降低计算量：

```bash
python src/detect_track.py --imgsz 320 --process-every 2 --show
```

测试视频文件：

```bash
python src/detect_track.py --source E:\test\video.mp4 --show
```

保存检测后的视频：

```bash
python src/detect_track.py --source E:\test\video.mp4 --save-video runs\track\out.mp4
```

不显示窗口，只保存报警日志。注意：`--show` 是开关参数，不写就是不显示窗口，所以后台运行直接：

```bash
python src/detect_track.py
```

## 参数含义

### 模型和输入

```text
--weights
```

人体检测权重路径。一般是训练好的 `best.pt`。

```text
--source
```

视频来源。

常见写法：

```text
0                  默认摄像头
E:\test\a.mp4      本地视频
rtsp://...         网络摄像头
```

```text
--imgsz
```

YOLO 输入尺寸。

常用：

```text
416  精度更稳
320  速度更快
```

```text
--device
```

运行设备。

常见写法：

```text
cpu
0
cuda:0
```

本地电脑有显卡时可用 `0` 或 `cuda:0`。

### 速度相关

```text
--process-every
```

每隔几帧检测一次。

```text
1  每帧都检测，精度和跟踪最好
2  每 2 帧检测一次，速度更快
3  每 3 帧检测一次，板端压力更小
```

```text
--camera-width
--camera-height
--camera-fps
```

设置摄像头采集分辨率和帧率。

例如：

```bash
--camera-width 640 --camera-height 480 --camera-fps 15
```

```text
--fps-log-interval
```

终端每隔几秒打印一次 FPS。

```text
--hide-fps
```

隐藏窗口左上角 FPS。

### 检测阈值

```text
--conf
```

人体检测置信度阈值。

```text
数值越高：误检更少，但可能漏检
数值越低：漏检更少，但可能误检
```

推荐：

```text
0.25
```

```text
--nms-iou
```

重叠框过滤阈值。

推荐：

```text
0.45
```

### 跟踪 ID 参数

```text
--track-thresh
```

高置信度检测框进入跟踪的阈值。

```text
--low-thresh
```

低置信度检测框补充匹配，用来减少遮挡后的 ID 丢失。

```text
--new-track-thresh
```

新建 ID 的最低置信度。

```text
--track-iou
```

检测框和历史轨迹的匹配阈值。

```text
--max-age
```

目标短暂消失后保留多少帧。

```text
--min-hits
```

目标出现几次后开始输出。

如果 ID 老是变，可以优先使用：

```bash
--track-thresh 0.25 --new-track-thresh 0.30 --track-iou 0.15 --max-age 90 --min-hits 1
```

### ROI 和警戒线

```text
--roi-file
```

ROI 配置文件。

推荐：

```bash
--roi-file configs/roi_example.json
```

```text
--roi
```

直接在命令里写 ROI。

示例：

```bash
--roi "220,160;460,160;460,360;220,360"
```

```text
--line
```

直接在命令里写警戒线。

示例：

```bash
--line "220,360;460,360"
```

```text
--line-direction
```

越线方向。

```text
any                   任意方向都报警
positive_to_negative  只检测一个方向
negative_to_positive  只检测反方向
```

### 异常行为参数

```text
--intrusion-min-frames
```

区域入侵需要连续进入 ROI 多少帧才报警。

默认：

```text
3
```

```text
--loiter-seconds
```

异常停留时间。

默认：

```text
20 秒
```

```text
--stationary-speed
```

异常停留的低速阈值。

默认：

```text
12 像素/秒
```

```text
--wandering-seconds
```

徘徊判断时间。

默认：

```text
30 秒
```

```text
--wandering-min-path
```

徘徊时累计移动路径至少多少像素。

```text
--wandering-max-displacement
```

徘徊时最终位移不能太大，否则认为是正常路过。

```text
--repeat-approach-window
```

多次靠近的统计时间窗口。

默认：

```text
60 秒
```

```text
--repeat-approach-count
```

时间窗口内进入 ROI 几次后报警。

默认：

```text
3 次
```

```text
--running-speed
```

快速奔跑速度阈值。

默认：

```text
200 像素/秒
```

如果走路也被误报成奔跑，可以调高：

```bash
--running-speed 260
```

```text
--running-min-frames
```

连续多少帧超过速度阈值才认为奔跑。

默认：

```text
8 帧
```

```text
--camera-block-ratio
```

人体框占画面比例超过多少，认为可能遮挡/贴近摄像头。

默认：

```text
0.55
```

意思是人体框面积超过画面 55%。

```text
--camera-block-seconds
```

遮挡持续多少秒后报警。

默认：

```text
3 秒
```

### 非工作时间报警

```text
--work-start
```

工作开始时间。

默认：

```text
08:00
```

```text
--work-end
```

工作结束时间。

默认：

```text
18:00
```

```text
--workdays
```

工作日。

默认：

```text
1,2,3,4,5
```

含义：

```text
1 周一
2 周二
3 周三
4 周四
5 周五
6 周六
7 周日
```

```text
--disable-after-hours
```

关闭非工作时间人员出现报警。

### 显示和上报

```text
--show
```

显示摄像头窗口。

不写 `--show` 就是不显示窗口。

```text
--english-ui
```

窗口 UI 使用英文。板端没有中文字体时可以用。

```text
--events-jsonl
```

保存报警日志。

默认：

```text
runs/track/events.jsonl
```

```text
--udp
```

UDP 上报报警。

示例：

```bash
--udp 192.168.1.100:9000
```

```text
--tcp
```

TCP 上报报警。

```text
--serial-port
--serial-baud
```

串口上报报警。

示例：

```bash
--serial-port COM3 --serial-baud 115200
```

板端可能是：

```bash
--serial-port /dev/ttyS3 --serial-baud 115200
```

```text
--summary-interval
```

人数统计上报间隔。

默认：

```text
1 秒
```

### 图像增强

```text
--enhance
```

开启弱光/逆光增强。

```text
--denoise
```

开启去噪，通常和 `--enhance` 一起用。

示例：

```bash
python src/detect_track.py --enhance --denoise --show
```

## 报警类型

当前支持：

```text
intrusion              区域入侵
loitering              异常停留
wandering              徘徊
repeat_approach        多次靠近敏感区域
line_crossing          穿越警戒线
after_hours_intrusion  非工作时间人员出现
camera_block           遮挡/贴近摄像头
running                快速奔跑
count                  人数统计
```

报警等级：

```text
warning   预警
critical  严重
```

示例：

```json
{"type":"intrusion","type_cn":"区域入侵","level":"critical","track_id":3,"time":12.44}
{"type":"loitering","type_cn":"异常停留","level":"warning","track_id":5,"duration":21.2}
{"type":"running","type_cn":"快速奔跑","level":"warning","track_id":2,"speed":230.5}
{"type":"count","type_cn":"人数统计","persons":2,"roi_persons":1}
```

## 文件作用

```text
configs/edge_runtime.yaml       运行参数配置
configs/roi_example.json        ROI 禁区和警戒线配置
src/detect_track.py             主运行文件
src/edge_person/config.py       配置文件读取、ROI/警戒线解析
src/edge_person/video.py        摄像头/视频输入、显示画布、坐标映射
src/edge_person/visual.py       检测框、状态栏、报警栏等画面绘制
src/edge_person/behavior.py     异常行为规则
src/edge_person/tracker.py      人员 ID 跟踪
src/edge_person/geometry.py     ROI、脚底中心点、越线等几何计算
src/edge_person/enhance.py      弱光、逆光、去噪增强
src/edge_person/reporter.py     JSONL、UDP、TCP、串口报警上报
runs/track/events.jsonl         运行后产生的报警日志
requirements.txt                Python 依赖
```

最常用的就是：

```text
src/detect_track.py
configs/edge_runtime.yaml
configs/roi_example.json
```
