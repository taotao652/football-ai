# Football AI ⚽

基于 YOLOv8 的足球球员检测 + 基于历史数据的比赛胜率预测。

---

## 📁 项目结构

- `yolo/` —— YOLOv8 球员检测模块
  - 检测图片/视频中的球员和足球
  - 支持推理、训练和结果可视化
  
- `predict/` —— 比赛胜率预测模块
  - 基于英超历史数据（2023-24 赛季）
  - 提取球队攻防特征，训练分类模型预测胜平负

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/taotao652/football-ai.git
cd football-ai

# 创建并激活虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装基础依赖
pip install ultralytics pandas scikit-learn球检测模型