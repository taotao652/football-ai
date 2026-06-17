# fetch_and_save.py
import requests
import json
import os

# 加载 .env 文件中的环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

API_KEY = os.getenv("API_FOOTBALL_KEY")
if not API_KEY:
    raise ValueError("请设置环境变量 API_FOOTBALL_KEY，或在项目根目录创建 .env 文件并写入 API_FOOTBALL_KEY=你的密钥")
url = "https://v3.football.api-sports.io/fixtures"
params = {"season": 2022, "league": 39, "status": "FT"}

response = requests.get(url, headers={"x-apisports-key": API_KEY}, params=params)
if response.status_code == 200:
    with open("data/premier_league_2022.json", "w") as f:
        json.dump(response.json(), f, indent=2)
    print("✅ 数据保存成功")