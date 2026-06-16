import json
import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.stats import poisson


def load_season(filepath):
    """加载一个赛季的数据"""
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
        })
    return pd.DataFrame(matches)


def compute_team_avg_goals(df):
    """计算每支球队的场均进球和失球"""
    team_stats = defaultdict(lambda: {'gs': 0, 'gc': 0, 'matches': 0})

    for _, row in df.iterrows():
        home, away = row['home'], row['away']
        team_stats[home]['gs'] += row['home_goals']
        team_stats[home]['gc'] += row['away_goals']
        team_stats[home]['matches'] += 1

        team_stats[away]['gs'] += row['away_goals']
        team_stats[away]['gc'] += row['home_goals']
        team_stats[away]['matches'] += 1

    # 计算场均
    for team in team_stats:
        if team_stats[team]['matches'] > 0:
            team_stats[team]['avg_scored'] = team_stats[team]['gs'] / team_stats[team]['matches']
            team_stats[team]['avg_conceded'] = team_stats[team]['gc'] / team_stats[team]['matches']
        else:
            team_stats[team]['avg_scored'] = 1.5
            team_stats[team]['avg_conceded'] = 1.5

    return team_stats


def predict_match(home_team, away_team, team_stats, n_sims=10000):
    """用泊松分布模拟比赛"""
    # 从字典中获取场均进球
    home_avg = team_stats[home_team]['avg_scored']
    away_avg = team_stats[away_team]['avg_scored']

    # 泊松分布模拟进球
    home_goals = np.random.poisson(home_avg, n_sims)
    away_goals = np.random.poisson(away_avg, n_sims)

    home_wins = np.sum(home_goals > away_goals)
    draws = np.sum(home_goals == away_goals)
    away_wins = np.sum(home_goals < away_goals)

    return {
        'home_win': home_wins / n_sims,
        'draw': draws / n_sims,
        'away_win': away_wins / n_sims
    }


def main():
    train_df = pd.concat([load_season(f'data/premier_league_202{i}.json') for i in [2, 3]], ignore_index=True)
    test_df = load_season('data/premier_league_2024.json')

    print(f"📊 训练: {len(train_df)}场, 测试: {len(test_df)}场")

    team_stats = compute_team_avg_goals(train_df)
    print(f"🔧 已计算 {len(team_stats)} 支球队的场均数据")

    correct = 0
    results = []

    for idx, row in test_df.iterrows():
        home = row['home']
        away = row['away']

        # 如果球队在训练集中没出现过，用联赛平均值
        if home not in team_stats:
            team_stats[home] = {'avg_scored': 1.5, 'avg_conceded': 1.5, 'gs': 0, 'gc': 0, 'matches': 0}
        if away not in team_stats:
            team_stats[away] = {'avg_scored': 1.5, 'avg_conceded': 1.5, 'gs': 0, 'gc': 0, 'matches': 0}

        prob = predict_match(home, away, team_stats)

        # 预测结果
        if prob['home_win'] >= prob['draw'] and prob['home_win'] >= prob['away_win']:
            pred = 0
        elif prob['draw'] >= prob['home_win'] and prob['draw'] >= prob['away_win']:
            pred = 1
        else:
            pred = 2

        # 实际结果
        if row['home_goals'] > row['away_goals']:
            actual = 0
        elif row['home_goals'] == row['away_goals']:
            actual = 1
        else:
            actual = 2

        if pred == actual:
            correct += 1

        results.append({
            'home': home,
            'away': away,
            'pred': pred,
            'actual': actual,
            'home_prob': prob['home_win'],
            'draw_prob': prob['draw'],
            'away_prob': prob['away_win']
        })

    print(f"\n✅ 泊松模拟预测准确率: {correct / len(test_df):.2%}")

    # 分析：当平局概率较高时的准确率
    high_draw = [r for r in results if r['draw_prob'] > 0.30]
    if high_draw:
        correct_high_draw = sum(1 for r in high_draw if r['pred'] == r['actual'])
        print(f"✅ 当平局概率 > 30% 时，准确率: {correct_high_draw / len(high_draw):.2%} ({len(high_draw)} 场)")


if __name__ == '__main__':
    main()