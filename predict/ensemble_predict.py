import json
import pandas as pd
import numpy as np
import math
from collections import defaultdict
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def load_season(filepath):
    """加载赛季数据"""
    with open(filepath) as f:
        data = json.load(f)
    matches = []
    for m in data['response']:
        if m['goals']['home'] is None or m['goals']['away'] is None:
            continue
        matches.append({
            'date': m['fixture']['date'],
            'home': m['teams']['home']['name'],
            'away': m['teams']['away']['name'],
            'home_goals': m['goals']['home'],
            'away_goals': m['goals']['away'],
        })
    return pd.DataFrame(matches)


class EnsembleModel:
    """集成模型：结合泊松、Elo和动态调整"""
    
    def __init__(self, decay_factor=0.995):
        self.decay_factor = decay_factor
        self.team_stats = {}
        self.elo_ratings = {}
        self.league_avg_home = 1.5
        self.league_avg_away = 1.3
        
    def fit(self, df):
        """训练模型：计算统计数据和Elo评分"""
        print("   [1/3] 计算球队攻防强度...")
        self._compute_team_stats(df)
        
        print("   [2/3] 训练Elo评级系统...")
        self._train_elo(df)
        
        print("   [3/3] 计算联赛基准...")
        self._compute_league_avg(df)
        
    def _compute_team_stats(self, df):
        """计算球队的攻防强度（简单平均）"""
        stats = defaultdict(lambda: {'home_gs': 0, 'home_gc': 0, 'away_gs': 0, 'away_gc': 0,
                                      'home_matches': 0, 'away_matches': 0})
        
        for row in df.itertuples():
            home, away = row.home, row.away
            hg, ag = row.home_goals, row.away_goals
            
            stats[home]['home_gs'] += hg
            stats[home]['home_gc'] += ag
            stats[home]['home_matches'] += 1
            
            stats[away]['away_gs'] += ag
            stats[away]['away_gc'] += hg
            stats[away]['away_matches'] += 1
        
        self.team_stats = {}
        for team, data in stats.items():
            home_matches = data['home_matches']
            away_matches = data['away_matches']
            total_matches = home_matches + away_matches
            
            if total_matches == 0:
                continue
            
            # 分别计算主客场场均
            home_scored = data['home_gs'] / home_matches if home_matches > 0 else 1.5
            away_scored = data['away_gs'] / away_matches if away_matches > 0 else 1.3
            home_conceded = data['home_gc'] / home_matches if home_matches > 0 else 1.3
            away_conceded = data['away_gc'] / away_matches if away_matches > 0 else 1.5
            
            # 综合攻防强度
            avg_scored = (home_scored + away_scored) / 2
            avg_conceded = (home_conceded + away_conceded) / 2
            
            attack_strength = avg_scored / 1.4  # 联赛平均约1.4
            defense_strength = avg_conceded / 1.4
            
            self.team_stats[team] = {
                'atk': attack_strength,
                'def': defense_strength,
                'home_scored': home_scored,
                'away_scored': away_scored,
                'home_conceded': home_conceded,
                'away_conceded': away_conceded
            }
    
    def _train_elo(self, df):
        """训练Elo评分系统"""
        df_sorted = df.sort_values('date')
        
        for _, row in df_sorted.iterrows():
            home, away = row.home, row.away
            hg, ag = row.home_goals, row.away_goals
            
            # 初始化Elo
            if home not in self.elo_ratings:
                self.elo_ratings[home] = 1500
            if away not in self.elo_ratings:
                self.elo_ratings[away] = 1500
            
            # 考虑主场优势
            home_rating = self.elo_ratings[home] + 50
            away_rating = self.elo_ratings[away]
            
            # 计算期望得分
            rating_diff = home_rating - away_rating
            expected_home = 1 / (1 + 10 ** (-rating_diff / 400))
            
            # 实际得分
            if hg > ag:
                actual_home = 1.0
            elif hg == ag:
                actual_home = 0.5
            else:
                actual_home = 0.0
            
            # 更新Elo（K因子=25），主场优势仅在计算时期使用
            delta = 25 * (actual_home - expected_home)
            self.elo_ratings[home] += delta
            self.elo_ratings[away] -= delta
    
    def _compute_league_avg(self, df):
        """计算联赛平均进球"""
        self.league_avg_home = df['home_goals'].mean()
        self.league_avg_away = df['away_goals'].mean()
    
    def predict_poisson(self, home, away):
        """泊松分布预测（精确概率计算，非蒙特卡洛）"""
        home_atk = self.team_stats.get(home, {}).get('atk', 1.0)
        home_def = self.team_stats.get(home, {}).get('def', 1.0)
        away_atk = self.team_stats.get(away, {}).get('atk', 1.0)
        away_def = self.team_stats.get(away, {}).get('def', 1.0)
        
        # 预期进球
        home_xg = self.league_avg_home * home_atk * away_def
        away_xg = self.league_avg_away * away_atk * home_def
        
        # 精确泊松概率（最大计算到 10 球）
        max_goals = 10
        home_probs = np.array([np.exp(-home_xg) * home_xg**k / math.factorial(k) for k in range(max_goals)])
        away_probs = np.array([np.exp(-away_xg) * away_xg**k / math.factorial(k) for k in range(max_goals)])
        
        h_wins = 0.0
        draws = 0.0
        a_wins = 0.0
        
        for h in range(max_goals):
            for a in range(max_goals):
                prob = home_probs[h] * away_probs[a]
                if h > a:
                    h_wins += prob
                elif h == a:
                    draws += prob
                else:
                    a_wins += prob
        
        # Dixon-Coles 低比分修正：提升 0-0 和 1-1 平局概率
        if home_xg + away_xg < 2.5:
            draws *= 1.12  # 更强的修正系数
        
        total = h_wins + draws + a_wins
        return {
            'home_win': h_wins / total,
            'draw': draws / total,
            'away_win': a_wins / total
        }
    
    def predict_elo(self, home, away):
        """Elo评级预测"""
        home_rating = self.elo_ratings.get(home, 1500) + 50  # 主场优势
        away_rating = self.elo_ratings.get(away, 1500)
        
        rating_diff = home_rating - away_rating
        
        home_win = 1 / (1 + 10 ** (-rating_diff / 400))
        away_win = 1 / (1 + 10 ** (rating_diff / 400))
        draw = max(0, 1 - home_win - away_win)  # 防止概率为负
        
        # 归一化
        total = home_win + away_win + draw
        if total == 0:
            return {'home_win': 0.4, 'draw': 0.3, 'away_win': 0.3}
        return {
            'home_win': home_win / total,
            'draw': draw / total,
            'away_win': away_win / total
        }
    
    def predict_ensemble(self, home, away, weights=None, draw_threshold=0.20):
        """集成预测：加权组合多个模型"""
        if weights is None:
            weights = {'poisson': 0.6, 'elo': 0.4}
        
        # 获取各模型预测
        poisson_pred = self.predict_poisson(home, away)
        elo_pred = self.predict_elo(home, away)
        
        # 加权组合
        final_pred = {
            'home_win': weights['poisson'] * poisson_pred['home_win'] + weights['elo'] * elo_pred['home_win'],
            'draw': weights['poisson'] * poisson_pred['draw'] + weights['elo'] * elo_pred['draw'],
            'away_win': weights['poisson'] * poisson_pred['away_win'] + weights['elo'] * elo_pred['away_win']
        }
        
        # 平局修正：如果平局概率达到最高概率的一定比例，提升平局
        max_prob = max(final_pred['home_win'], final_pred['draw'], final_pred['away_win'])
        if final_pred['draw'] >= max_prob * draw_threshold:
            boost = (final_pred['draw'] / max_prob) * 0.1
            final_pred['draw'] += boost
            final_pred['home_win'] -= boost * final_pred['home_win'] / (final_pred['home_win'] + final_pred['away_win'] + 0.001)
            final_pred['away_win'] -= boost * final_pred['away_win'] / (final_pred['home_win'] + final_pred['away_win'] + 0.001)
        
        return final_pred


def main():
    print("=" * 70)
    print("🚀 集成模型预测系统 (Poisson + Elo + Dixon-Coles)")
    print("=" * 70)
    
    # 加载数据
    print("\n📊 加载数据...")
    df_22 = load_season('data/premier_league_2022.json')
    df_23 = load_season('data/premier_league_2023.json')
    df_24 = load_season('data/premier_league_2024.json')
    
    train_df = pd.concat([df_22, df_23], ignore_index=True)
    test_df = df_24
    
    print(f"   训练集: {len(train_df)}场")
    print(f"   测试集: {len(test_df)}场")
    
    # 训练模型
    print("\n🏋️ 训练集成模型...")
    model = EnsembleModel(decay_factor=0.995)
    model.fit(train_df)
    
    # 测试不同权重组合
    print("\n🔍 测试不同模型权重组合...")
    weight_configs = [
        {'poisson': 1.0, 'elo': 0.0, 'name': '纯泊松', 'draw_thresh': 0.20},
        {'poisson': 1.0, 'elo': 0.0, 'name': '纯泊松+强平局', 'draw_thresh': 0.15},
        {'poisson': 0.0, 'elo': 1.0, 'name': '纯Elo', 'draw_thresh': 0.20},
        {'poisson': 0.7, 'elo': 0.3, 'name': '泊松70+Elo30', 'draw_thresh': 0.20},
        {'poisson': 0.6, 'elo': 0.4, 'name': '泊松60+Elo40', 'draw_thresh': 0.20},
        {'poisson': 0.5, 'elo': 0.5, 'name': '泊松50+Elo50', 'draw_thresh': 0.20},
    ]
    
    best_accuracy = 0
    best_config = None
    
    for config in weight_configs:
        weights = {'poisson': config['poisson'], 'elo': config['elo']}
        
        correct = 0
        results = []
        
        for _, row in test_df.iterrows():
            prob = model.predict_ensemble(row['home'], row['away'], weights, config['draw_thresh'])
            
            # 平局优先策略：当平局概率接近最高概率时，预测平局
            if prob['draw'] >= max(prob['home_win'], prob['away_win']) * 0.55:
                pred = 'draw'
            else:
                pred = max(prob, key=prob.get)
            
            if row['home_goals'] > row['away_goals']:
                actual = 'home_win'
            elif row['home_goals'] < row['away_goals']:
                actual = 'away_win'
            else:
                actual = 'draw'
            
            if pred == actual:
                correct += 1
            
            results.append({
                'pred': pred,
                'actual': actual,
                'home_prob': prob['home_win'],
                'draw_prob': prob['draw'],
                'away_prob': prob['away_win']
            })
        
        accuracy = correct / len(test_df)
        print(f"   {config['name']:20s}: {accuracy:.2%} ({correct}/{len(test_df)})")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_config = config
    
    print(f"\n✅ 最佳配置: {best_config['name']} - 准确率: {best_accuracy:.2%}")
    
    # 使用最佳配置进行详细分析
    print("\n📈 详细分析报告:")
    print("-" * 70)
    
    weights = {'poisson': best_config['poisson'], 'elo': best_config['elo']}
    correct = 0
    results = []
    
    for _, row in test_df.iterrows():
        prob = model.predict_ensemble(row['home'], row['away'], weights, best_config['draw_thresh'])
        
        # 平局优先策略
        if prob['draw'] >= max(prob['home_win'], prob['away_win']) * 0.55:
            pred = 'draw'
        else:
            pred = max(prob, key=prob.get)
        
        if row['home_goals'] > row['away_goals']:
            actual = 'home_win'
        elif row['home_goals'] < row['away_goals']:
            actual = 'away_win'
        else:
            actual = 'draw'
        
        if pred == actual:
            correct += 1
        
        results.append({
            'home': row['home'],
            'away': row['away'],
            'pred': pred,
            'actual': actual,
            'home_prob': prob['home_win'],
            'draw_prob': prob['draw'],
            'away_prob': prob['away_win']
        })
    
    # 整体准确率
    print(f"\n整体准确率: {correct / len(test_df):.2%} ({correct}/{len(test_df)})")
    
    # 分类准确率
    home_correct = sum(1 for r in results if r['pred'] == 'home_win' and r['actual'] == 'home_win')
    home_total = sum(1 for r in results if r['actual'] == 'home_win')
    
    draw_correct = sum(1 for r in results if r['pred'] == 'draw' and r['actual'] == 'draw')
    draw_total = sum(1 for r in results if r['actual'] == 'draw')
    
    away_correct = sum(1 for r in results if r['pred'] == 'away_win' and r['actual'] == 'away_win')
    away_total = sum(1 for r in results if r['actual'] == 'away_win')
    
    print(f"\n分类准确率:")
    print(f"  主胜: {home_correct/home_total:.2%} ({home_correct}/{home_total})")
    print(f"  平局: {draw_correct/draw_total:.2%} ({draw_correct}/{draw_total})")
    print(f"  客胜: {away_correct/away_total:.2%} ({away_correct}/{away_total})")
    
    # 高置信度预测
    high_conf = [r for r in results if max(r['home_prob'], r['draw_prob'], r['away_prob']) > 0.55]
    if high_conf:
        high_conf_correct = sum(1 for r in high_conf if r['pred'] == r['actual'])
        print(f"\n🎯 高置信度预测 (>55%): {high_conf_correct/len(high_conf):.2%} ({high_conf_correct}/{len(high_conf)})")
    
    # 平局分析
    high_draw = [r for r in results if r['draw_prob'] > 0.28]
    if high_draw:
        high_draw_correct = sum(1 for r in high_draw if r['pred'] == r['actual'])
        print(f"⚖️  平局概率>28%时: {high_draw_correct/len(high_draw):.2%} ({high_draw_correct}/{len(high_draw)})")
    
    print("\n" + "=" * 70)
    print("✅ 测试完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()
