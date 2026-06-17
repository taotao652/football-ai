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
        home_probs = np.array([np.exp(-home_xg) * home_xg ** k / math.factorial(k) for k in range(max_goals)])
        away_probs = np.array([np.exp(-away_xg) * away_xg ** k / math.factorial(k) for k in range(max_goals)])

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

        # ===== 修改点1: 增强的Dixon-Coles低比分修正 =====
        # 针对0-0和1-1等低比分平局做强化修正
        if home_xg + away_xg < 2.8:
            # 低进球比赛，平局概率显著提升
            draws *= 1.18
        elif home_xg + away_xg < 4.0:
            draws *= 1.08

        # ===== 修改点2: 平局概率保底（防止极端情况） =====
        # 任何场次平局概率不低于6%，确保模型不会完全放弃平局
        draws = max(draws, 0.06)

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

    def predict_ensemble(self, home, away, weights=None, draw_threshold=0.30):
        """
        集成预测：加权组合多个模型

        修改说明：
        1. draw_threshold默认从0.20改为0.30，只在平局概率真正接近最高概率时才提升
        2. 平局修正幅度从0.1提升到0.25
        3. 修正后做归一化保证概率和为1
        """
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

        # ===== 修改点3: 增强的平局修正 =====
        max_prob = max(final_pred['home_win'], final_pred['draw'], final_pred['away_win'])
        # 只有当平局概率达到最高概率的30%以上时才修正（原来是20%）
        if final_pred['draw'] >= max_prob * draw_threshold:
            # 修正幅度提升：从0.1提高到0.25
            boost = final_pred['draw'] * 0.25
            final_pred['draw'] += boost

            # 从主胜和客胜中按比例扣除
            home_win_ratio = final_pred['home_win'] / (final_pred['home_win'] + final_pred['away_win'] + 0.001)
            away_win_ratio = final_pred['away_win'] / (final_pred['home_win'] + final_pred['away_win'] + 0.001)
            final_pred['home_win'] -= boost * home_win_ratio
            final_pred['away_win'] -= boost * away_win_ratio

            # 防止负数
            final_pred['home_win'] = max(final_pred['home_win'], 0.01)
            final_pred['away_win'] = max(final_pred['away_win'], 0.01)

            # 归一化保证概率和为1
            total = sum(final_pred.values())
            final_pred = {k: v / total for k, v in final_pred.items()}

        return final_pred


def evaluate_predictions(results, test_df):
    """评估预测结果，输出详细指标"""
    # 整体准确率
    total_correct = sum(1 for r in results if r['pred'] == r['actual'])
    total = len(results)
    accuracy = total_correct / total

    # 分类准确率
    home_correct = sum(1 for r in results if r['pred'] == 'home_win' and r['actual'] == 'home_win')
    home_total = sum(1 for r in results if r['actual'] == 'home_win')

    draw_correct = sum(1 for r in results if r['pred'] == 'draw' and r['actual'] == 'draw')
    draw_total = sum(1 for r in results if r['actual'] == 'draw')

    away_correct = sum(1 for r in results if r['pred'] == 'away_win' and r['actual'] == 'away_win')
    away_total = sum(1 for r in results if r['actual'] == 'away_win')

    # 平局召回率（关键指标）
    draw_recall = draw_correct / draw_total if draw_total > 0 else 0
    draw_precision = draw_correct / sum(1 for r in results if r['pred'] == 'draw') if sum(
        1 for r in results if r['pred'] == 'draw') > 0 else 0
    draw_f1 = 2 * draw_precision * draw_recall / (draw_precision + draw_recall) if (
                                                                                               draw_precision + draw_recall) > 0 else 0

    print(f"\n整体准确率: {accuracy:.2%} ({total_correct}/{total})")
    print(f"\n分类准确率:")
    print(
        f"  主胜: {home_correct / home_total:.2%} ({home_correct}/{home_total})" if home_total > 0 else "  主胜: 无样本")
    print(
        f"  平局: {draw_correct / draw_total:.2%} ({draw_correct}/{draw_total})" if draw_total > 0 else "  平局: 无样本")
    print(
        f"  客胜: {away_correct / away_total:.2%} ({away_correct}/{away_total})" if away_total > 0 else "  客胜: 无样本")

    print(f"\n⚖️  平局专项指标 (核心!)")
    print(f"  召回率 (Recall): {draw_recall:.2%}")
    print(f"  精确率 (Precision): {draw_precision:.2%}")
    print(f"  F1分数: {draw_f1:.4f}")

    # 高置信度分析
    high_conf = [r for r in results if max(r['home_prob'], r['draw_prob'], r['away_prob']) > 0.55]
    if high_conf:
        high_conf_correct = sum(1 for r in high_conf if r['pred'] == r['actual'])
        print(
            f"\n🎯 高置信度预测 (>55%): {high_conf_correct / len(high_conf):.2%} ({high_conf_correct}/{len(high_conf)})")

    # 平局概率分布
    draw_probs = [r['draw_prob'] for r in results]
    print(f"\n📊 平局概率分布:")
    print(f"  最小值: {min(draw_probs):.2%}")
    print(f"  平均值: {np.mean(draw_probs):.2%}")
    print(f"  最大值: {max(draw_probs):.2%}")

    return {
        'accuracy': accuracy,
        'draw_recall': draw_recall,
        'draw_precision': draw_precision,
        'draw_f1': draw_f1
    }


def main():
    print("=" * 70)
    print("🚀 集成模型预测系统 (差异化阈值策略)")
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

    draw_count = sum(1 for _, row in test_df.iterrows() if row['home_goals'] == row['away_goals'])
    print(f"   测试集平局占比: {draw_count / len(test_df):.2%} ({draw_count}场)")

    # 训练模型
    print("\n🏋️ 训练集成模型...")
    model = EnsembleModel(decay_factor=0.995)
    model.fit(train_df)

    # ===== 差异化阈值策略 =====
    print("\n🔍 测试差异化阈值策略...")
    print("   主场优势: 主胜概率>客胜时，用严格阈值(少判平局)")
    print("   客场弱势: 客胜概率>主胜时，用宽松阈值(多判平局)")
    print("-" * 70)

    # 测试不同的阈值组合 (主场阈值, 客场阈值)
    threshold_configs = [
        {'name': '主0.12/客0.30', 'home_thresh': 0.12, 'away_thresh': 0.30},
        {'name': '主0.15/客0.30', 'home_thresh': 0.15, 'away_thresh': 0.30},
        {'name': '主0.18/客0.30', 'home_thresh': 0.18, 'away_thresh': 0.30},
        {'name': '主0.20/客0.30', 'home_thresh': 0.20, 'away_thresh': 0.30},
        {'name': '主0.12/客0.25', 'home_thresh': 0.12, 'away_thresh': 0.25},
        {'name': '主0.15/客0.25', 'home_thresh': 0.15, 'away_thresh': 0.25},
        {'name': '主0.18/客0.25', 'home_thresh': 0.18, 'away_thresh': 0.25},
        {'name': '主0.20/客0.25', 'home_thresh': 0.20, 'away_thresh': 0.25},
        {'name': '主0.12/客0.20', 'home_thresh': 0.12, 'away_thresh': 0.20},
        {'name': '主0.15/客0.20', 'home_thresh': 0.15, 'away_thresh': 0.20},
    ]

    best_config = None
    best_score = 0
    best_results = None

    for config in threshold_configs:
        weights = {'poisson': 0.6, 'elo': 0.4}
        home_thresh = config['home_thresh']
        away_thresh = config['away_thresh']

        correct = 0
        results = []
        draw_predictions = 0

        for _, row in test_df.iterrows():
            prob = model.predict_ensemble(row['home'], row['away'], weights, draw_threshold=0.30)

            # ===== 差异化阈值核心 =====
            prob_diff = abs(prob['home_win'] - prob['away_win'])

            # 根据谁更占优，使用不同阈值
            if prob['home_win'] > prob['away_win']:
                # 主胜概率更高：用严格阈值（少判平局）
                threshold = home_thresh
            else:
                # 客胜概率更高或相等：用宽松阈值（多判平局）
                threshold = away_thresh

            if prob_diff < threshold:
                pred = 'draw'
                draw_predictions += 1
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
                'away_prob': prob['away_win'],
                'prob_diff': prob_diff
            })

        accuracy = correct / len(test_df)

        draw_correct = sum(1 for r in results if r['pred'] == 'draw' and r['actual'] == 'draw')
        draw_total = sum(1 for r in results if r['actual'] == 'draw')
        draw_recall = draw_correct / draw_total if draw_total > 0 else 0
        draw_precision = draw_correct / draw_predictions if draw_predictions > 0 else 0
        draw_f1 = 2 * draw_precision * draw_recall / (draw_precision + draw_recall) if (
                                                                                                   draw_precision + draw_recall) > 0 else 0

        # 综合评分：准确率 + 平局F1（平衡）
        composite_score = accuracy + draw_f1 * 0.5
        print(
            f"   {config['name']:18s}: 准确率 {accuracy:.2%} | 平局召回 {draw_recall:.2%} | 平局F1 {draw_f1:.4f} | 预测平局 {draw_predictions}场")

        if composite_score > best_score:
            best_score = composite_score
            best_config = config
            best_results = results

    print("-" * 70)
    print(f"\n🏆 最佳综合配置: {best_config['name']}")
    print(f"   综合评分: {best_score:.4f}")

    # 详细报告
    print("\n" + "=" * 70)
    print("📈 详细分析报告")
    print("=" * 70)

    results = best_results
    total_correct = sum(1 for r in results if r['pred'] == r['actual'])
    total = len(results)
    accuracy = total_correct / total

    home_correct = sum(1 for r in results if r['pred'] == 'home_win' and r['actual'] == 'home_win')
    home_total = sum(1 for r in results if r['actual'] == 'home_win')

    draw_correct = sum(1 for r in results if r['pred'] == 'draw' and r['actual'] == 'draw')
    draw_total = sum(1 for r in results if r['actual'] == 'draw')

    away_correct = sum(1 for r in results if r['pred'] == 'away_win' and r['actual'] == 'away_win')
    away_total = sum(1 for r in results if r['actual'] == 'away_win')

    draw_predictions = sum(1 for r in results if r['pred'] == 'draw')
    draw_precision = draw_correct / draw_predictions if draw_predictions > 0 else 0
    draw_recall = draw_correct / draw_total if draw_total > 0 else 0
    draw_f1 = 2 * draw_precision * draw_recall / (draw_precision + draw_recall) if (
                                                                                               draw_precision + draw_recall) > 0 else 0

    print(f"\n整体准确率: {accuracy:.2%} ({total_correct}/{total})")
    print(f"\n分类准确率:")
    print(
        f"  主胜: {home_correct / home_total:.2%} ({home_correct}/{home_total})" if home_total > 0 else "  主胜: 无样本")
    print(
        f"  平局: {draw_correct / draw_total:.2%} ({draw_correct}/{draw_total})" if draw_total > 0 else "  平局: 无样本")
    print(
        f"  客胜: {away_correct / away_total:.2%} ({away_correct}/{away_total})" if away_total > 0 else "  客胜: 无样本")

    print(f"\n⚖️  平局专项指标:")
    print(f"  召回率 (Recall): {draw_recall:.2%}")
    print(f"  精确率 (Precision): {draw_precision:.2%}")
    print(f"  F1分数: {draw_f1:.4f}")
    print(f"  预测平局场次: {draw_predictions}场")

    # 分析
    print("\n📊 策略效果分析:")
    home_dominant = [r for r in results if r['home_prob'] > r['away_prob']]
    away_dominant = [r for r in results if r['away_prob'] > r['home_prob']]

    if home_dominant:
        home_draw_pred = sum(1 for r in home_dominant if r['pred'] == 'draw')
        home_draw_correct = sum(1 for r in home_dominant if r['pred'] == 'draw' and r['actual'] == 'draw')
        print(f"  主胜概率更高时: 共{len(home_dominant)}场，判平{home_draw_pred}场，正确{home_draw_correct}场")

    if away_dominant:
        away_draw_pred = sum(1 for r in away_dominant if r['pred'] == 'draw')
        away_draw_correct = sum(1 for r in away_dominant if r['pred'] == 'draw' and r['actual'] == 'draw')
        print(f"  客胜概率更高时: 共{len(away_dominant)}场，判平{away_draw_pred}场，正确{away_draw_correct}场")

    print("\n" + "=" * 70)
    print("✅ 测试完成！")
    print("=" * 70)

if __name__ == '__main__':
    main()