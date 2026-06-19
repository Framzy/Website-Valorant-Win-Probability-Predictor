# === train_general.py ===
# Pre-compute weighted scores, popular comps, dan thresholds
# untuk Casual mode (scoring heuristik, TANPA JST model)
import pandas as pd
import numpy as np
import joblib
import math
from collections import defaultdict, Counter

print("[INFO] Loading dataset...")
df = pd.read_csv("valorant_dataset_all.csv")

# === 1. Aggregate per-match ===
print("[INFO] Aggregating per-match data...")
group_cols = ['Tournament', 'Stage', 'Match Type', 'Map', 'Team']
df_g = df.groupby(group_cols).agg({
    'Agent': lambda x: list(x),
    'Total Wins By Map': 'first',
    'Total Loss By Map': 'first'
}).reset_index()

df_g = df_g[df_g['Agent'].apply(len) == 5].reset_index(drop=True)
df_g['TP'] = df_g['Total Wins By Map'] + df_g['Total Loss By Map']
df_g['WR'] = df_g['Total Wins By Map'] / df_g['TP']
print(f"[INFO] Total matches (5-agent): {len(df_g)}")

# === 2. Agent-Role Map ===
AGENT_ROLE_MAP = {
    # Duelist
    'iso': 'duelist', 'jett': 'duelist', 'raze': 'duelist',
    'reyna': 'duelist', 'yoru': 'duelist', 'neon': 'duelist',
    'phoenix': 'duelist', 'waylay': 'duelist',
    # Initiator
    'skye': 'initiator', 'sova': 'initiator', 'breach': 'initiator',
    'fade': 'initiator', 'kayo': 'initiator', 'gekko': 'initiator',
    'tejo': 'initiator',
    # Controller
    'brimstone': 'controller', 'omen': 'controller', 'viper': 'controller',
    'astra': 'controller', 'harbor': 'controller', 'clove': 'controller',
    # Sentinel
    'killjoy': 'sentinel', 'cypher': 'sentinel', 'sage': 'sentinel',
    'chamber': 'sentinel', 'deadlock': 'sentinel', 'vyse': 'sentinel',
}
ROLE_ORDER = ["duelist", "initiator", "controller", "sentinel"]

# === 3. Per-Agent-Per-Map Stats ===
print("[INFO] Computing per-agent-per-map statistics...")
agent_map_raw = defaultdict(lambda: {'count': 0, 'wr_sum': 0.0})
for _, row in df_g.iterrows():
    for agent in row['Agent']:
        key = (agent, row['Map'])
        agent_map_raw[key]['count'] += 1
        agent_map_raw[key]['wr_sum'] += row['WR']

# Max pick count per map (untuk normalisasi confidence)
maps = df_g['Map'].unique().tolist()
map_max_count = {}
for m in maps:
    counts = [v['count'] for k, v in agent_map_raw.items() if k[1] == m]
    map_max_count[m] = max(counts) if counts else 1

# Compute weighted scores
agent_map_weighted = {}
for (agent, map_name), stats in agent_map_raw.items():
    avg_wr = stats['wr_sum'] / stats['count']
    max_cnt = map_max_count.get(map_name, 1)
    confidence = math.log(1 + stats['count']) / math.log(1 + max_cnt)
    weighted = avg_wr * confidence
    agent_map_weighted[(agent, map_name)] = {
        'count': stats['count'],
        'avg_wr': round(avg_wr, 4),
        'confidence': round(confidence, 4),
        'weighted': round(weighted, 4),
    }

print(f"[INFO] Agent-map combos computed: {len(agent_map_weighted)}")

# === 4. Role-Popular Agents per Map ===
# Dict: {(role, map): [(agent, count), ...]} sorted by count desc
print("[INFO] Computing role-popular agents per map...")
role_popular = defaultdict(list)
for (agent, map_name), stats in agent_map_weighted.items():
    role = AGENT_ROLE_MAP.get(agent, None)
    if role:
        role_popular[(role, map_name)].append((agent, stats['count']))

# Sort by count descending
for key in role_popular:
    role_popular[key] = sorted(role_popular[key], key=lambda x: -x[1])

# === 5. Popular Comps per Map (Top 3) ===
print("[INFO] Computing popular comps per map...")
popular_comps = {}
for m in maps:
    df_map = df_g[df_g['Map'] == m]
    combos = []
    for _, row in df_map.iterrows():
        combo = tuple(sorted(row['Agent']))
        combos.append((combo, row['WR']))

    combo_stats = defaultdict(lambda: {'count': 0, 'wr_sum': 0.0})
    for combo, wr in combos:
        combo_stats[combo]['count'] += 1
        combo_stats[combo]['wr_sum'] += wr

    top = sorted(combo_stats.items(), key=lambda x: -x[1]['count'])[:3]
    popular_comps[m] = [
        {
            "agents": list(combo),
            "count": stats['count'],
            "avg_wr": round(stats['wr_sum'] / stats['count'] * 100, 1),
        }
        for combo, stats in top
    ]

# === 6. Gauge Thresholds (dari distribusi weighted scores aktual) ===
print("[INFO] Computing gauge thresholds from actual match distributions...")
all_match_scores = []
for _, row in df_g.iterrows():
    map_name = row['Map']
    match_scores = []
    for agent in row['Agent']:
        data = agent_map_weighted.get((agent, map_name))
        if data:
            match_scores.append(data['weighted'])
        else:
            match_scores.append(0.0)
    all_match_scores.append(np.mean(match_scores))

arr = np.array(all_match_scores)
casual_thresholds = {
    'p25': round(float(np.percentile(arr, 25)), 4),
    'p50': round(float(np.percentile(arr, 50)), 4),
    'p75': round(float(np.percentile(arr, 75)), 4),
    'min': round(float(arr.min()), 4),
    'max': round(float(arr.max()), 4),
}
print(f"[INFO] Thresholds: P25={casual_thresholds['p25']:.3f} "
      f"P50={casual_thresholds['p50']:.3f} P75={casual_thresholds['p75']:.3f}")
print(f"[INFO] Range: [{casual_thresholds['min']:.3f}, {casual_thresholds['max']:.3f}]")

# === 7. Save All Artifacts ===
print("\n[INFO] Saving artifacts...")

joblib.dump(dict(agent_map_weighted), "agent_map_weighted.pkl")
print("  -> agent_map_weighted.pkl")

joblib.dump(dict(map_max_count), "map_max_count.pkl")
print("  -> map_max_count.pkl")

joblib.dump(dict(role_popular), "role_popular_agents.pkl")
print("  -> role_popular_agents.pkl")

joblib.dump(popular_comps, "popular_comps.pkl")
print("  -> popular_comps.pkl")

joblib.dump(casual_thresholds, "casual_thresholds.pkl")
print("  -> casual_thresholds.pkl")

joblib.dump(AGENT_ROLE_MAP, "agent_role_map_general.pkl")
print("  -> agent_role_map_general.pkl")

# === 8. Summary ===
print("\n[INFO] Done! Summary:")
print(f"  Maps: {len(maps)}")
print(f"  Agent-map combos: {len(agent_map_weighted)}")
print(f"  Popular comps: {sum(len(v) for v in popular_comps.values())} entries across {len(popular_comps)} maps")
print(f"  Gauge thresholds: Rendah < {casual_thresholds['p25']*100:.1f}% | "
      f"Moderat < {casual_thresholds['p50']*100:.1f}% | "
      f"Tinggi < {casual_thresholds['p75']*100:.1f}% | "
      f"Sangat Tinggi >= {casual_thresholds['p75']*100:.1f}%")

# Show top agent per role per map as verification
print("\n[INFO] Top agents per role (sample: Lotus):")
for role in ROLE_ORDER:
    agents = role_popular.get((role, 'Lotus'), [])
    top3 = agents[:3]
    names = [f"{a}({c})" for a, c in top3]
    print(f"  {role:12s}: {', '.join(names)}")
