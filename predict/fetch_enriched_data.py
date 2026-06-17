import requests
import json
import time
import os
from datetime import datetime

# 加载 .env 文件中的环境变量（从项目根目录）
try:
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    load_dotenv(dotenv_path=env_path)
except ImportError:
    pass

API_KEY = os.getenv("API_FOOTBALL_KEY")
if not API_KEY:
    raise ValueError("请设置环境变量 API_FOOTBALL_KEY，或在项目根目录创建 .env 文件并写入 API_FOOTBALL_KEY=你的密钥")
BASE_URL = "https://v3.football.api-sports.io"

headers = {
    'x-rapidapi-key': API_KEY,
    'x-rapidapi-host': 'v3.football.api-sports.io'
}


def get_fixture_statistics(fixture_id):
    """获取单场比赛的详细统计数据
    
    返回:
        dict | None: 成功返回统计数据字典，失败返回 None
        str: 失败原因 ('quota_exceeded', 'http_error', 'network_error')
    """
    url = f"{BASE_URL}/fixtures/statistics"
    params = {'fixture': fixture_id}
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        data = response.json()
        
        if response.status_code == 200:
            if data['response'] and len(data['response']) == 2:
                return {
                    'home': extract_team_stats(data['response'][0]),
                    'away': extract_team_stats(data['response'][1])
                }
            else:
                print(f"  ⚠️  无数据 (可能有统计但格式异常)")
                return None
        
        # 检查是否是额度耗尽
        if response.status_code == 429:
            print(f"  🚫 额度已用完")
            return 'QUOTA_EXCEEDED'
        
        # 检查 response 中的错误信息
        errors = data.get('errors', {})
        if errors:
            error_msg = str(errors.get('requests', errors))[:80]
            print(f"  ⚠️  API 错误: {error_msg}")
            # 额度相关错误
            if 'quota' in error_msg.lower() or 'limit' in error_msg.lower() or 'rate' in error_msg.lower():
                return 'QUOTA_EXCEEDED'
        else:
            print(f"  ⚠️  HTTP {response.status_code}: {str(data)[:80]}")
            
    except requests.exceptions.Timeout:
        print(f"  ⏱️  请求超时")
    except requests.exceptions.ConnectionError:
        print(f"  🔌 网络连接失败")
    except Exception as e:
        print(f"  ❌ {type(e).__name__}: {e}")
    
    return None


def extract_team_stats(team_data):
    """从球队统计数据中提取关键特征"""
    stats_dict = {}
    for stat in team_data['statistics']:
        stat_type = stat['type']
        value = stat['value']
        
        # 转换值为数字（有些是字符串）
        if value is not None:
            try:
                value = float(value) if '.' in str(value) else int(value)
            except:
                value = 0
        
        stats_dict[stat_type] = value
    
    # 提取关键特征
    return {
        'shots_on_goal': stats_dict.get('Shots on Goal', 0),
        'shots_off_goal': stats_dict.get('Shots off Goal', 0),
        'total_shots': stats_dict.get('Total Shots', 0),
        'blocked_shots': stats_dict.get('Blocked Shots', 0),
        'shots_insidebox': stats_dict.get('Shots insidebox', 0),
        'shots_outsidebox': stats_dict.get('Shots outsidebox', 0),
        'possession': stats_dict.get('Ball Possession', 50),  # 默认50%
        'corners': stats_dict.get('Corner Kicks', 0),
        'fouls': stats_dict.get('Fouls', 0),
        'yellow_cards': stats_dict.get('Yellow Cards', 0),
        'red_cards': stats_dict.get('Red Cards', 0),
        'offsides': stats_dict.get('Offsides', 0),
        'passes_total': stats_dict.get('Total passes', 0),
        'passes_accurate': stats_dict.get('Passes accurate', 0),
        'pass_accuracy': stats_dict.get('Passes %', 0),
        'tackles': stats_dict.get('Tackles', 0),
        'saves': stats_dict.get('Goalkeeper saves', 0),
    }


def enrich_match_data(base_matches, output_file, start_index=0, max_requests=None):
    """为已有比赛数据添加详细统计特征
    
    参数:
        base_matches: 基础比赛数据列表
        output_file: 输出文件路径
        start_index: 从第几场开始（断点续传）
        max_requests: 最多请求多少次（默认 None = 不限）
    """
    print(f"🚀 开始获取 {len(base_matches)} 场比赛的详细统计数据...")
    print(f"   从第 {start_index + 1} 场开始")
    if max_requests:
        print(f"   本次最多请求 {max_requests} 次")
    print()
    
    # 加载已有进度（如果有的话）
    enriched_matches = []
    if os.path.exists(output_file) and start_index > 0:
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                enriched_matches = json.load(f)
        except Exception:
            pass
    
    consecutive_failures = 0
    request_count = 0
    
    try:
        for i, match in enumerate(base_matches):
            if i < start_index:
                continue
            
            # 检查是否达到本次请求上限
            if max_requests and request_count >= max_requests:
                print(f"\n⏸️  已达到本次请求上限 ({max_requests} 次)，停止")
                break
                
            fixture_id = match.get('fixture', {}).get('id')
            if not fixture_id:
                print(f"⚠️  跳过第 {i+1} 场：缺少 fixture ID")
                continue
            
            print(f"[{i+1}/{len(base_matches)}] 获取比赛 {fixture_id} 的统计数据...", end=' ')
            request_count += 1
            
            # 获取统计数据
            stats = get_fixture_statistics(fixture_id)
            
            if stats == 'QUOTA_EXCEEDED':
                print(f"\n\n⚠️  API 额度已用完！")
                print(f"   已成功获取 {len(enriched_matches)} 场")
                print(f"   剩余 {len(base_matches) - i} 场可下次继续")
                print(f"   下次从第 {i + 1} 场开始 (start_index={i})")
                break
            
            if stats and stats != 'QUOTA_EXCEEDED':
                # 合并基础数据和统计数据
                enriched_match = {
                    'fixture_id': fixture_id,
                    'date': match['fixture']['date'],
                    'home_team': match['teams']['home']['name'],
                    'away_team': match['teams']['away']['name'],
                    'home_goals': match['goals']['home'],
                    'away_goals': match['goals']['away'],
                    'home_halftime_goals': match['score']['halftime']['home'],
                    'away_halftime_goals': match['score']['halftime']['away'],
                }
                
                # 添加主队统计特征
                for key, value in stats['home'].items():
                    enriched_match[f'home_{key}'] = value
                
                # 添加客队统计特征
                for key, value in stats['away'].items():
                    enriched_match[f'away_{key}'] = value
                
                enriched_matches.append(enriched_match)
                consecutive_failures = 0
                print("✅")
                
                # 每成功一场就保存，确保中断不丢数据
                save_progress(enriched_matches, output_file)
            else:
                consecutive_failures += 1
                print("❌ 失败")
                
                # 连续失败 10 次，可能是网络问题或 API 挂了
                if consecutive_failures >= 10:
                    print(f"\n⚠️  连续失败 {consecutive_failures} 次，可能网络异常，停止")
                    break
            
            # API 限制：每秒最多1个请求
            time.sleep(1.1)
            
            # 每50场打印一次里程碑
            if (i + 1) % 50 == 0:
                print(f"\n💾 里程碑: 已完成 {len(enriched_matches)} 场比赛\n")
    
    except KeyboardInterrupt:
        print(f"\n\n⚠️  用户中断，已保存 {len(enriched_matches)} 场进度")
    
    if enriched_matches:
        save_progress(enriched_matches, output_file)
    print(f"\n✅ 完成！共获取 {len(enriched_matches)} 场比赛的完整数据")
    
    return enriched_matches


def save_progress(matches, output_file):
    """保存进度到文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(matches, f, ensure_ascii=False, indent=2)


def load_base_matches(filepath):
    """加载基础比赛数据"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['response']


def main():
    # 选择要处理的赛季（免费API仅22赛季有统计数据，23/24需付费）
    seasons = [
        ('data/premier_league_2022.json', 'data/enriched_2022.json'),
        # ('data/premier_league_2023.json', 'data/enriched_2023.json'),  # 免费套餐无数据
        # ('data/premier_league_2024.json', 'data/enriched_2024.json'),  # 免费套餐无数据
    ]
    
    for base_file, output_file in seasons:
        if not os.path.exists(base_file):
            print(f"⚠️  文件不存在: {base_file}")
            continue
        
        # 加载基础数据
        base_matches = load_base_matches(base_file)
        total = len(base_matches)
        
        # 检查是否有未完成的进度文件 → 断点续传
        start_index = 0
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
                if existing:
                    last_id = existing[-1].get('fixture_id')
                    for idx, m in enumerate(base_matches):
                        if m.get('fixture', {}).get('id') == last_id:
                            start_index = idx + 1
                            break
                    print(f"📂 检测到已有进度: {len(existing)}/{total} 场")
                    if start_index >= total:
                        print(f"✅ 已全部完成，跳过\n")
                        continue
                    print(f"   从第 {start_index + 1} 场继续\n")
            except Exception as e:
                print(f"⚠️  读取进度文件失败: {e}，重新开始\n")
                start_index = 0
        
        print(f"\n{'='*60}")
        print(f"📊 处理赛季: {base_file}")
        print(f"{'='*60}")
        print(f"共 {total} 场，已完成 {start_index} 场，剩余 {total - start_index} 场")
        print()
        
        # 获取详细统计数据
        enriched_matches = enrich_match_data(base_matches, output_file, start_index)
        
        print(f"\n{'='*60}")
        print(f"✅ {base_file} 处理完成！")
        print(f"{'='*60}\n")
        
        # 显示示例数据
        if enriched_matches:
            print("📋 示例数据结构:")
            sample = enriched_matches[0]
            for key in sorted(sample.keys()):
                print(f"  - {key}: {sample[key]}")
            print()


if __name__ == '__main__':
    print("=" * 60)
    print("🚀 API-FOOTBALL 数据采集器")
    print("=" * 60)
    print("\n⚠️  注意：")
    print("  - 每场比赛需要 ~1.1 秒")
    print("  - 380场比赛约需 7-8 分钟")
    print("  - 免费 API 有每日配额限制，用完自动停止")
    print("  - 免费套餐仅支持 2022 赛季统计数据")
    print("\n💡 使用技巧：")
    print("  - 额度用完会自动停止并保存进度")
    print("  - 第二天直接再次运行，自动从上次中断处续传")
    print("  - 按 Ctrl+C 可以随时中断，进度会自动保存\n")
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断，进度已保存")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
