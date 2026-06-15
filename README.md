# Football AI - 足球检测项目

使用 YOLOv8 检测足球比赛中的球员和足球。

## 功能
- 实时检测球员和足球
- 支持图片、视频、摄像头输入
- 训练好的模型可直接使用

## 模型效果
![检测示例](examples/test.jpg)

## 快速开始
```bash
# 安装依赖
pip install -r requirements.txt

# 运行检测
yolo predict model=models/best.pt source=your_image.jpg
