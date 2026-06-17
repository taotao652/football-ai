# YOLO 球员检测模块 ⚽

基于 YOLOv8 的足球及球员目标检测，支持推理和训练。

---

## 📁 目录结构

```
yolo/
├── config/
│   └── football.yaml          # 训练配置（2类：足球、球员）
├── data/
│   └── person_yolo_dataset/   # 标注数据集（图片 + 标签）
├── models/
│   ├── best-v1.0.pt           # 自定义训练的最佳模型
│   └── yolov8n.pt             # YOLOv8 nano 基础权重
├── examples/                  # 测试图片
├── runs/detect/               # 训练/推理输出
└── scripts/
    └── voc2yolo_converter.py  # VOC → YOLO 格式转换工具
```

---

## 🚀 快速使用

### 推理

```bash
# 使用预训练模型检测单张图片
yolo predict model=models/best-v1.0.pt source=examples/test.jpg

# 检测视频
yolo predict model=models/best-v1.0.pt source=your_video.mp4
```

结果保存在 `runs/detect/predict/` 目录。

### 训练

```bash
yolo train model=yolov8n.pt data=config/football.yaml epochs=100
```

### 模型信息

| 模型 | 说明 |
|------|------|
| `best-v1.0.pt` | 自定义训练，检测 football + person |
| `yolov8n.pt` | YOLOv8 nano，基础预训练权重 |

---

## 🏷️ 标签类别

| ID | 类别 |
|----|------|
| 0 | football（足球） |
| 1 | person（球员） |

---

## 🔧 环境依赖

```bash
pip install ultralytics
```
