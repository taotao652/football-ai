# Football AI ⚽

基于 YOLOv8 的足球球员检测 + 基于历史数据的比赛胜率预测。

---

## 📁 项目结构

- `yolo/` —— YOLOv8 球员检测模块
  - 检测图片/视频中的球员和足球
  - 支持推理、训练和结果可视化
  
- `predict/` —— 比赛胜率预测模块
  - 基于英超 2022 赛季历史数据
  - 集成泊松分布 + Elo 评级 + XGBoost 多种模型
  - 采集详细比赛统计特征（射门、控球、传球等）

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

# 安装依赖
pip install ultralytics pandas scikit-learn xgboost requests
```

### 2. 配置 API Key（数据采集需要）

```bash
# 在项目根目录创建 .env 文件
echo "API_FOOTBALL_KEY=你的密钥" > .env
```

### 3. 运行预测

```bash
cd predict
python ensemble_predict.py   # 集成模型（推荐）
python poisson_predict.py    # 基础泊松模型
```

### 4. 采集比赛统计数据（可选，需要 API 配额）

```bash
cd predict
python fetch_enriched_data.py  # 自动续传，额度用完自动停止
```

---

## 📊 当前状态

- 数据采集：2022 赛季进行中（免费 API 每日约 10 次请求）
- 模型准确率：集成模型 ~51.6%（详见 `predict/MODEL_README.md`）
- YOLO 检测：预训练模型 `yolo/models/best-v1.0.pt`