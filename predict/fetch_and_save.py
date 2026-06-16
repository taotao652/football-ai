# fetch_and_save.py
import requests
import json

API_KEY = "abdf27ac7228d75ec554b96cb4bb5b03"
url = "https://v3.football.api-sports.io/fixtures"
params = {"season": 2022, "league": 39, "status": "FT"}

response = requests.get(url, headers={"x-apisports-key": API_KEY}, params=params)
if response.status_code == 200:
    with open("data/premier_league_2022.json", "w") as f:
        json.dump(response.json(), f, indent=2)
    print("✅ 数据保存成功")