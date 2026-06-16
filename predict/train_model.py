import json
import pandas as pd
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def load_season(filepath):
    with open(filepath) as f:
        data = json.load(f)
    matches = []
    for m in data['response']:
        if m['goals']['home'] is None or m['goals']['away'] is None:
            continue
        matches.append({
            'home': m['teams']['home']['name'],
            'away': m['teams']['away']['name'],
            'home_goals': m['goals']['home'],
            'away_goals': m['goals']['away'],
            'result': 0 if m['goals']['home'] > m['goals']['away'] else (1 if m['goals']['home'] == m['goals']['away'] else 2)
        })
    return pd.DataFrame(matches)

def build_features(df):
    team_stats = defaultdict(lambda: {'gs': 0, 'gc': 0, 'matches': 0, 'wins': 0})
    X, y = [], []
    for _, row in df.iterrows():
        home, away = row['home'], row['away']
        h = team_stats[home]
        a = team_stats[away]
        home_avg_scored = h['gs'] / h['matches'] if h['matches'] > 0 else 1.5
        home_avg_conceded = h['gc'] / h['matches'] if h['matches'] > 0 else 1.5
        home_win_rate = h['wins'] / h['matches'] if h['matches'] > 0 else 0.4
        away_avg_scored = a['gs'] / a['matches'] if a['matches'] > 0 else 1.5
        away_avg_conceded = a['gc'] / a['matches'] if a['matches'] > 0 else 1.5
        away_win_rate = a['wins'] / a['matches'] if a['matches'] > 0 else 0.4
        X.append([home_avg_scored, home_avg_conceded, home_win_rate, away_avg_scored, away_avg_conceded, away_win_rate])
        y.append(row['result'])
        h['gs'] += row['home_goals']
        h['gc'] += row['away_goals']
        h['matches'] += 1
        h['wins'] += 1 if row['result'] == 0 else 0
        a['gs'] += row['away_goals']
        a['gc'] += row['home_goals']
        a['matches'] += 1
        a['wins'] += 1 if row['result'] == 2 else 0
    return X, y

def main():
    train_df = pd.concat([load_season(f'data/premier_league_202{i}.json') for i in [2, 3]], ignore_index=True)
    test_df = load_season('data/premier_league_2024.json')
    print(f"📊 训练: {len(train_df)}场, 测试: {len(test_df)}场")
    X_train, y_train = build_features(train_df)
    X_test, y_test = build_features(test_df)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"✅ 随机森林准确率: {accuracy_score(y_test, y_pred):.2%}")
    print(classification_report(y_test, y_pred, target_names=['主胜', '平局', '客胜']))

if __name__ == '__main__':
    main()

